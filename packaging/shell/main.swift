// DiffusionBear native shell.
//
// A thin Swift/WKWebView window around the FastAPI backend, so the app is one
// click: no Terminal, no Vite dev server, no Node. The backend serves the
// production SPA from the same origin, so this is one process and one port.
//
// Deliberate choices:
//   * The backend is spawned WITHOUT --reload. Reload restarts the worker on
//     every file change and is a dev-only affordance.
//   * PYTHONHOME is never set. It leaks into every grandchild process and kills
//     interpreters that are not ours (the SDXL and Qwen engines are spawned as
//     grandchildren and have their own Python). PYTHONPATH is not needed either,
//     because the bundled venv is invoked as `-m uvicorn`.
//   * The venv is launched via `<venv>/bin/python -m ...`, never via a console
//     script. Console-script shebangs break under a path containing a space, and
//     macOS truncates shebangs at the first space -- that is how a whole app ends
//     up booting under a stale interpreter.
//   * The 28 GB model/LoRA/gallery store is never inside the bundle. It lives in
//     Application Support and is passed through MLX_DIFFUSION_ASSET_DIR.

import AppKit
import WebKit

// MARK: - Configuration

private let backendPort = 8001
// Generous on purpose. A cold start imports mflux, MLX and torch from inside a
// 1.8 GB bundle, and on a busy 16 GB machine that measured ~95s -- uncomfortably
// close to a 120s ceiling, which produced a spurious "backend did not start" while
// the server was mid-import. The window shows progress, so waiting costs nothing.
private let pollInterval: TimeInterval = 0.4

private struct Paths {
    /// Contents/Resources, derived from this executable's own location rather than
    /// Bundle.main.resourceURL. Bundle's resource resolution is not reliable here:
    /// it returned the .app root instead of Contents/Resources, which left the
    /// backend's cwd pointing at the bundle root so uvicorn had no `main:app` to
    /// import and the app hung for minutes with an empty log. Walking up from the
    /// executable is unambiguous: MacOS/MLX-Diffusion -> MacOS -> Contents.
    static let resources: URL = {
        let exe = Bundle.main.executableURL
            ?? URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL
        return exe
            .deletingLastPathComponent()   // MacOS
            .deletingLastPathComponent()   // Contents
            .appendingPathComponent("Resources", isDirectory: true)
    }()
    static let backend: URL = resources.appendingPathComponent("backend", isDirectory: true)
    /// Layout inside Resources is load-bearing, not cosmetic. generator.py resolves
    /// both engine interpreters relative to the repo root:
    ///     sdxl:  <root>/venv-sdxl/bin/python
    ///     qwen:  <root>/venv/bin/python   (falls back to sys.executable)
    /// and main.py serves the SPA from <root>/frontend/dist. Placing the runtime at
    /// exactly these names means the bundle needs no code change and no env-var
    /// overrides to find its engines.
    static let runtime: URL = resources.appendingPathComponent("venv", isDirectory: true)
    static let sdxl: URL = resources.appendingPathComponent("venv-sdxl", isDirectory: true)
    static var python: URL { runtime.appendingPathComponent("bin/python") }
    static var sdxlPython: URL { sdxl.appendingPathComponent("bin/python") }

    /// External, user-owned store. Not bundled, never copied.
    ///
    /// Renamed from "MLX-Diffusion" to "DiffusionBear", but the model store can be
    /// tens of gigabytes and lives on an external volume, so it is never moved or
    /// recreated. If the new directory does not exist but the old one does, use the
    /// old one and say so in the log. Copying would waste the user 28 GB of disk to
    /// rename an app.
    static let supportNames = ["DiffusionBear", "MLX-Diffusion"]
    static var support: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        for name in supportNames {
            let candidate = base.appendingPathComponent(name, isDirectory: true)
            if FileManager.default.fileExists(atPath: candidate.appendingPathComponent("data").path) {
                if name != supportNames[0] {
                    NSLog("DiffusionBear: reusing existing data at %@", candidate.path)
                }
                return candidate
            }
        }
        return base.appendingPathComponent(supportNames[0], isDirectory: true)
    }
    static var assetDir: URL { support.appendingPathComponent("data", isDirectory: true) }
    static var logFile: URL {
        let logs = support.appendingPathComponent("Logs", isDirectory: true)
        try? FileManager.default.createDirectory(at: logs, withIntermediateDirectories: true)
        return logs.appendingPathComponent("backend.log")
    }
}

private var backendURL: URL { URL(string: "http://127.0.0.1:\(backendPort)")! }

/// True when the app itself lives on a mounted volume rather than the system disk.
///
/// This matters more than it looks. macOS gates access to removable and external
/// volumes behind a TCC consent prompt. If DiffusionBear is launched from
/// /Volumes/Externe and nobody is at the machine to click Allow, the backend's
/// very first open() of its own bundled stdlib blocks forever at 0% CPU -- the app
/// appears to hang with an empty log. That is exactly what was measured, and it is
/// why the app belongs in /Applications.
private var isOnExternalVolume: Bool {
    Paths.resources.path.hasPrefix("/Volumes/")
}

private var externalVolumeName: String {
    let parts = Paths.resources.path.split(separator: "/")
    return parts.count > 1 ? "/Volumes/\(parts[1])" : "/Volumes"
}

/// Append to the launch log. Separate from the child's own log so a child's output
/// buffering can never hide the launch record.
private func trace(_ message: String) {
    let stamp = ISO8601DateFormatter().string(from: Date())
    let line = "\(stamp) [pid \(ProcessInfo.processInfo.processIdentifier)] \(message)\n"
    NSLog("%@", message)
    let url = Paths.support
        .appendingPathComponent("Logs", isDirectory: true)
        .appendingPathComponent("launch.log")
    try? FileManager.default.createDirectory(
        at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    // FileHandle(forWritingTo:) throws when the file is absent, so create it first.
    if !FileManager.default.fileExists(atPath: url.path) {
        FileManager.default.createFile(atPath: url.path, contents: nil)
    }
    if let data = line.data(using: .utf8),
       let handle = try? FileHandle(forWritingTo: url) {
        defer { try? handle.close() }
        handle.seekToEndOfFile()
        handle.write(data)
    }
}

// MARK: - Backend supervision

/// Owns the backend child process: starts it, waits for it to answer, watches it,
/// and makes sure it is gone when the app quits.
final class BackendProcess {
    private var process: Process?
    private var logHandle: FileHandle?
    private var pollTimer: Timer?
    private var attempt = 0
    private var startedAt = Date()
    private(set) var isReady = false
    var onReady: (() -> Void)?
    var onFailure: ((String) -> Void)?

    /// The child occasionally blocks in open() at 0% CPU during interpreter
    /// startup -- it has been seen with only dyld mapped, no stdlib, for over ten
    /// minutes, and it is not reproducible from a shell. Importing the same bundled
    /// interpreter directly always works, and so does relaunching the same
    /// untouched bundle moments later, so the pragmatic answer is to treat it as
    /// transient and retry rather than to declare failure. Three attempts, then
    /// give up and say so.
    private let maxAttempts = 3
    private let attemptTimeout: TimeInterval = 100

    deinit { stop() }


    /// Refuse to start rather than fight an existing listener: two backends on one
    /// port means the UI silently talks to someone else's server.
    static func portInUse(_ port: Int) -> Bool {
        // lsof is present on every macOS install; no added dependency.
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/sbin/lsof")
        p.arguments = ["-nP", "-iTCP:\(port)", "-sTCP:LISTEN"]
        p.standardOutput = FileHandle.nullDevice
        p.standardError = FileHandle.nullDevice
        do { try p.run(); p.waitUntilExit(); return p.terminationStatus == 0 }
        catch { return false }
    }

    func start() {
        // Fail fast and loudly on a broken bundle. Without this the app appears to
        // hang: uvicorn is launched with a bad cwd, finds no `main:app`, and the
        // process sits there while the 120s startup timer runs.
        var problems: [String] = []
        if !FileManager.default.isExecutableFile(atPath: Paths.python.path) {
            problems.append("bundled Python missing:\n  \(Paths.python.path)")
        }
        if !FileManager.default.fileExists(atPath: Paths.backend.appendingPathComponent("main.py").path) {
            problems.append("backend missing:\n  \(Paths.backend.appendingPathComponent("main.py").path)")
        }
        if !FileManager.default.fileExists(
            atPath: Paths.resources.appendingPathComponent("frontend/dist/index.html").path) {
            problems.append("built SPA missing:\n  \(Paths.resources.appendingPathComponent("frontend/dist/index.html").path)")
        }
        if !problems.isEmpty {
            onFailure?("This app bundle looks incomplete.\n\n" + problems.joined(separator: "\n")
                + "\n\nResolved resources directory:\n  \(Paths.resources.path)")
            return
        }
        if !FileManager.default.isExecutableFile(atPath: Paths.sdxlPython.path) {
            NSLog("DiffusionBear: SDXL runtime missing at %@ -- that engine will be unavailable",
                  Paths.sdxlPython.path)
        }

        try? FileManager.default.createDirectory(at: Paths.support, withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(at: Paths.assetDir, withIntermediateDirectories: true)

        // Truncate through the same handle rather than a second open of the file, so
        // there is only ever one writer on it.
        if !FileManager.default.fileExists(atPath: Paths.logFile.path) {
            FileManager.default.createFile(atPath: Paths.logFile.path, contents: nil)
        }
        logHandle = try? FileHandle(forWritingTo: Paths.logFile)
        try? logHandle?.truncate(atOffset: 0)
        try? logHandle?.seek(toOffset: 0)

        let p = Process()
        p.executableURL = Paths.python
        // -u: Python block-buffers stdout when it is not a tty, so uvicorn's
        // "Started server process" line sat in a 4-8 KB buffer and the log stayed
        // empty while the server was perfectly healthy -- which reads as a failed
        // start. Unbuffered output makes the log the honest place to look.
        p.arguments = ["-u", "-m", "uvicorn", "main:app",
                       "--host", "127.0.0.1",
                       "--port", String(backendPort),
                       "--no-access-log"]
        p.currentDirectoryURL = Paths.backend

        var env = ProcessInfo.processInfo.environment
        // Both are needed. ASSET_DIR alone only relocates the model store, while
        // DATA_DIR (gallery, settings.json, LoRA registry, uploads, pid files) is
        // separate -- without it the app writes into the signed bundle, which grows
        // without bound and invalidates the bundle's own signature on next launch.
        env["MLX_DIFFUSION_ASSET_DIR"] = Paths.assetDir.path
        env["MLX_DIFFUSION_DATA_DIR"] = Paths.assetDir.path
        // PYTHONHOME is intentionally absent: it leaks into every grandchild and
        // kills interpreters that are not ours, and the SDXL and Qwen engines are
        // spawned as grandchildren with their own interpreters. The venvs are
        // invoked as `-m uvicorn`, so no PYTHONPATH is needed either.
        env.removeValue(forKey: "PYTHONHOME")
        p.environment = env

        p.standardOutput = logHandle
        p.standardError = logHandle

        do {
            try p.run()
        } catch {
            onFailure?("Could not start the backend:\n\(error.localizedDescription)")
            return
        }
        process = p
        attempt += 1
        startedAt = Date()
        trace("attempt \(attempt)/\(maxAttempts): backend pid \(p.processIdentifier) cwd \(Paths.backend.path)")

        let thisAttempt = attempt
        pollTimer = Timer.scheduledTimer(withTimeInterval: pollInterval, repeats: true) { [weak self] timer in
            guard let self else { timer.invalidate(); return }
            if p.isRunning == false {
                timer.invalidate()
                trace("attempt \(thisAttempt): child exited, status \(p.terminationStatus) reason \(p.terminationReason.rawValue)")
                self.retryOrFail("The backend exited during startup (status \(p.terminationStatus)).")
                return
            }
            if Self.answers(path: "/api/version") {
                timer.invalidate()
                self.isReady = true
                trace("ready after \(String(format: "%.1f", Date().timeIntervalSince(self.startedAt)))s on attempt \(thisAttempt)")
                self.onReady?()
                return
            }
            if Date().timeIntervalSince(startedAt) > attemptTimeout {
                timer.invalidate()
                trace("attempt \(thisAttempt): no response after \(Int(attemptTimeout))s, killing pid \(p.processIdentifier)")
                self.retryOrFail("The backend did not respond within \(Int(attemptTimeout))s.")
            }
        }
    }

    /// Kill whatever is running and try again, up to `maxAttempts`.
    private func retryOrFail(_ reason: String) {
        stop()
        isReady = false
        if attempt < maxAttempts {
            trace("retrying: \(reason)")
            // A short pause: a hung child may still be releasing its file handles.
            DispatchQueue.main.asyncAfter(deadline: .now() + 2) { [weak self] in self?.start() }
        } else {
            onFailure?("""
                \(reason)

                If a macOS permission dialog is open right now — "DiffusionBear would \
                like to access files in your <volume>" — that is the cause. It blocks the \
                backend until answered, and nothing else will tell you so.

                This app is running from \(Paths.resources.path), which is on \(externalVolumeName). \
                macOS asks for consent on external volumes, and an unanswered prompt looks \
                exactly like a hung app. Move DiffusionBear.app into /Applications to avoid \
                the prompt entirely.

                Tried \(maxAttempts) times. Logs:

                  backend: \(Paths.logFile.path)
                  launch:  \(Paths.support.appendingPathComponent("Logs/launch.log").path)
                """)
        }
    }

    /// Liveness probe. Any HTTP answer at all means uvicorn is serving, which is
    /// all we need to know -- a 404 or 405 still proves the socket is up. Requiring
    /// a specific 200 from a HEAD here is what broke the first launch: FastAPI
    /// registers only GET for a @app.get route, so a HEAD probe can never return
    /// 200 and the app reported "backend did not start" while it was serving fine.
    private final class Probe: @unchecked Sendable {
        private let lock = NSLock()
        private var value = false
        var ok: Bool {
            get { lock.lock(); defer { lock.unlock() }; return value }
            set { lock.lock(); value = newValue; lock.unlock() }
        }
    }

    private static func answers(path: String) -> Bool {
        var request = URLRequest(url: backendURL.appendingPathComponent(path))
        request.httpMethod = "GET"
        request.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        request.timeoutInterval = pollInterval + 0.5
        let done = DispatchSemaphore(value: 0)
        let box = Probe()
        URLSession.shared.dataTask(with: request) { _, response, error in
            box.ok = (error == nil && response != nil)
            done.signal()
        }.resume()
        _ = done.wait(timeout: .now() + 2)
        return box.ok
    }

    func stop() {
        pollTimer?.invalidate(); pollTimer = nil
        guard let p = process, p.isRunning else { return }
        p.terminate()
        // Give uvicorn a moment to shut down its workers, then insist.
        let deadline = Date().addingTimeInterval(8)
        while p.isRunning && Date() < deadline {
            usleep(150_000)
        }
        if p.isRunning {
            kill(p.processIdentifier, SIGKILL)
        }
        try? logHandle?.close()
        logHandle = nil
        process = nil
    }
}

// MARK: - Menus

/// Hosts the web view and contributes text-editing actions to the context menu.
///
/// The bare shell sets no NSMenu at all, which is why copy/paste did not work:
/// with no Edit menu there is nothing for ⌘C/⌘V to route to, and WebKit's own
/// context menu has no Cut/Copy/Paste for editable fields to fall back on.
///
/// This *appends* to whatever menu WebKit builds rather than replacing it, so
/// link and image context items keep working. Each item has a nil target, so the
/// responder chain decides whether it is enabled -- the actions are greyed out
/// when the click was not in a text field.
final class AppWebView: WKWebView {
    override func menu(for event: NSEvent) -> NSMenu? {
        let menu = super.menu(for: event) ?? NSMenu(title: "")
        let alreadyThere = Set(menu.items.map(\.title))
        let wanted: [(String, Selector, String)] = [
            ("Cut", #selector(NSText.cut(_:)), "x"),
            ("Copy", #selector(NSText.copy(_:)), "c"),
            ("Paste", #selector(NSText.paste(_:)), "v"),
            ("Select All", #selector(NSText.selectAll(_:)), "a"),
        ].filter { !alreadyThere.contains($0.0) }
        guard !wanted.isEmpty else { return menu }
        menu.addItem(.separator())
        for (title, action, key) in wanted {
            let item = NSMenuItem(title: title, action: action, keyEquivalent: key)
            item.target = nil
            menu.addItem(item)
        }
        return menu
    }
}

/// The app menu bar. Without this there is no Edit menu, which is the root cause
/// of copy/paste not working -- ⌘C/⌘V/⌘A and the Edit menu both need somewhere to
/// send their actions.
private func installMainMenu() {
    let mainMenu = NSMenu()

    // Application menu (DiffusionBear ▸ About/Quit)
    let appItem = NSMenuItem()
    mainMenu.addItem(appItem)
    let appMenu = NSMenu(title: "DiffusionBear")
    appMenu.addItem(withTitle: "About DiffusionBear", action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)), keyEquivalent: "")
    appMenu.addItem(.separator())
    appMenu.addItem(withTitle: "Hide DiffusionBear", action: #selector(NSApplication.hide(_:)), keyEquivalent: "h")
    let hideOthers = appMenu.addItem(withTitle: "Hide Others", action: #selector(NSApplication.hideOtherApplications(_:)), keyEquivalent: "h")
    hideOthers.keyEquivalentModifierMask = [.command, .option]
    appMenu.addItem(withTitle: "Show All", action: #selector(NSApplication.unhideAllApplications(_:)), keyEquivalent: "")
    appMenu.addItem(.separator())
    appMenu.addItem(withTitle: "Quit DiffusionBear", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
    appItem.submenu = appMenu

    // Edit menu -- this is what makes copy/paste work at all.
    let editItem = NSMenuItem()
    mainMenu.addItem(editItem)
    let editMenu = NSMenu(title: "Edit")
    editMenu.addItem(withTitle: "Undo", action: Selector(("undo:")), keyEquivalent: "z")
    let redo = editMenu.addItem(withTitle: "Redo", action: Selector(("redo:")), keyEquivalent: "z")
    redo.keyEquivalentModifierMask = [.command, .shift]
    editMenu.addItem(.separator())
    editMenu.addItem(withTitle: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
    editMenu.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
    editMenu.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
    editMenu.addItem(withTitle: "Paste and Match Style", action: Selector(("pasteAsPlainText:")), keyEquivalent: "v")
    editMenu.addItem(withTitle: "Delete", action: #selector(NSText.delete(_:)), keyEquivalent: "")
    editMenu.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
    editItem.submenu = editMenu

    // View
    let viewItem = NSMenuItem()
    mainMenu.addItem(viewItem)
    let viewMenu = NSMenu(title: "View")
    viewMenu.addItem(withTitle: "Reload Interface", action: Selector(("reloadIgnoringCache:")), keyEquivalent: "r")
    viewMenu.addItem(.separator())
    viewMenu.addItem(withTitle: "Enter Full Screen", action: #selector(NSWindow.toggleFullScreen(_:)), keyEquivalent: "f")
    viewMenu.addItem(withTitle: "Actual Size", action: Selector(("zoom:")), keyEquivalent: "0")
    viewItem.submenu = viewMenu

    // Window
    let windowItem = NSMenuItem()
    mainMenu.addItem(windowItem)
    let windowMenu = NSMenu(title: "Window")
    windowMenu.addItem(withTitle: "Minimize", action: #selector(NSWindow.performMiniaturize(_:)), keyEquivalent: "m")
    windowMenu.addItem(withTitle: "Zoom", action: #selector(NSWindow.performZoom(_:)), keyEquivalent: "")
    windowMenu.addItem(.separator())
    windowMenu.addItem(withTitle: "Bring All to Front", action: #selector(NSApplication.arrangeInFront(_:)), keyEquivalent: "")
    windowItem.submenu = windowMenu
    NSApp.windowsMenu = windowMenu

    NSApp.mainMenu = mainMenu
}

// MARK: - UI

final class AppDelegate: NSObject, NSApplicationDelegate, WKNavigationDelegate {
    private var window: NSWindow!
    private var webView: WKWebView!
    private let backend = BackendProcess()
    private var statusLabel: NSTextField!
    private var activityToken: NSObjectProtocol?

    /// Drop cached responses for the local UI, keeping cookies and localStorage.
    ///
    /// The backend now serves index.html with no-store, but a copy cached before
    /// that fix can still be in the persistent data store, and WKWebView will happily
    /// render last build's JavaScript against this build's API -- which is exactly the
    /// "Load failed, 0 images" state, with every endpoint returning 200. Only the
    /// caches are removed; localStorage holds user preferences and image tags, so it
    /// must survive.
    private func purgeInterfaceCache() {
        // Only the caches. WKWebsiteDataStore exposes allWebsiteDataTypes() but not
        // per-cache-type constants, so build the removal set by subtraction and keep
        // what holds user state: localStorage (preferences, image tags), session
        // storage, cookies and IndexedDB.
        let available = WKWebsiteDataStore.allWebsiteDataTypes()
        let keep = Set<String>(
            [WKWebsiteDataTypeLocalStorage,
             WKWebsiteDataTypeSessionStorage,
             WKWebsiteDataTypeCookies,
             WKWebsiteDataTypeIndexedDBDatabases]
                .filter { available.contains($0) }
        )
        let types = WKWebsiteDataStore.allWebsiteDataTypes().subtracting(keep)
        guard !types.isEmpty else { return }
        WKWebsiteDataStore.default().removeData(ofTypes: types, modifiedSince: .distantPast) {
            trace("purged \(types.count) web view cache type(s); kept localStorage, cookies and IndexedDB")
        }
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        installMainMenu()
        purgeInterfaceCache()
        // Hold a user-initiated activity for the app's whole life. Without this,
        // macOS is free to App-Nap this process (and its backend child) whenever
        // the window is occluded or the app is not frontmost, which stalls the
        // model and makes a launch look like a hang.
        activityToken = ProcessInfo.processInfo.beginActivity(
            options: [.userInitiated, .idleSystemSleepDisabled],
            reason: "Local image generation backend is running")
        buildWindow()

        if BackendProcess.portInUse(backendPort) {
            showBlockingError(
                title: "Port \(backendPort) is already in use",
                message: """
                Another process is already listening on 127.0.0.1:\(backendPort).

                Stop it, or restart this app, then try again. Running two backends \
                on one port would leave the window showing someone else's server.
                """)
            return
        }

        setStatus("Starting the local engine…", spinner: true)
        if isOnExternalVolume {
            // Ask for the volume up front and say why. A blocked prompt is
            // indistinguishable from a hung app otherwise.
            NSApp.activate(ignoringOtherApps: true)
            setStatus("""
                Waiting for macOS permission to read \(externalVolumeName).
                Click Allow if a dialog appeared — DiffusionBear cannot start until you do.
                """, spinner: true)
            requestAccessToOwnBundle()
        }
        backend.onReady = { [weak self] in
            guard let self else { return }
            self.setStatus(nil)
            self.webView.load(URLRequest(url: backendURL))
            NSApp.activate(ignoringOtherApps: true)
        }
        backend.onFailure = { [weak self] message in
            self?.showBlockingError(title: "DiffusionBear could not start", message: message)
        }
        backend.start()
    }

    func applicationWillTerminate(_ notification: Notification) {
        // Stop the child first: the SDXL and Qwen engines are its grandchildren and
        // hold ~10 GB each, so an orphaned backend would leak them.
        backend.stop()
        if let token = activityToken {
            ProcessInfo.processInfo.endActivity(token)
            activityToken = nil
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    // MARK: window

    private func buildWindow() {
        let config = WKWebViewConfiguration()
        config.websiteDataStore = .default()   // keep logins/session across launches
        webView = AppWebView(frame: NSRect(x: 0, y: 0, width: 1280, height: 840), configuration: config)
        webView.navigationDelegate = self
        webView.autoresizingMask = [.width, .height]

        let content = NSView(frame: NSRect(x: 0, y: 0, width: 1280, height: 840))
        // Give the web view a real frame. It used to be created with .zero and only
        // given an autoresizing mask, which left it zero-sized forever -- the window
        // opened and the page loaded, but nothing was ever drawn. autoresizing
        // adjusts relative to the current frame, so it can never rescue a zero one.
        webView.frame = content.bounds
        content.addSubview(webView)

        statusLabel = NSTextField(labelWithString: "")
        statusLabel.font = .systemFont(ofSize: 13, weight: .medium)
        statusLabel.textColor = .secondaryLabelColor
        statusLabel.alignment = .center

        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1280, height: 840),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered, defer: false)
        window.title = "DiffusionBear"
        // The window was un-draggable: titlebarAppearsTransparent + hidden title
        // left nothing that looked like a title bar to grab. Dragging the window
        // background is the fix that works regardless of how the title bar is drawn,
        // and it is the behaviour every borderless-ish window should have.
        window.isMovableByWindowBackground = true
        window.isReleasedWhenClosed = false
        window.minSize = NSSize(width: 900, height: 600)
        window.contentView = content
        window.center()
        window.makeKeyAndOrderFront(nil)
        window.makeFirstResponder(webView)
    }


    /// Touch one file inside the bundle so macOS raises the volume's consent prompt
    /// now, while the status line can explain it, instead of surfacing it as a
    /// mystery block inside the child process later.
    ///
    /// Note the path: Info.plist lives at Contents/Info.plist, not under Resources,
    /// so probing Resources/Info.plist silently returned early and the whole check
    /// was a no-op. Probe main.py -- a file the backend genuinely needs, in the
    /// directory it genuinely runs from.
    private func requestAccessToOwnBundle() {
        let probe = Paths.backend.appendingPathComponent("main.py")
        guard let data = try? Data(contentsOf: probe) else {
            trace("volume probe FAILED for \(probe.path)")
            return
        }
        trace("volume probe ok: read \(data.count) bytes from \(probe.path) on \(externalVolumeName)")
    }

    private func setStatus(_ text: String?, spinner: Bool = false) {
        guard let window else { return }
        if let text {
            statusLabel.stringValue = text
            statusLabel.isHidden = false
            // Size to the text: the volume-permission hint is several lines and a
            // 20pt single-line field would clip it.
            let width = window.contentView?.bounds.width ?? 1280
            let fitting = (text as NSString).boundingRect(
                with: NSSize(width: width - 80, height: .greatestFiniteMagnitude),
                options: [.usesLineFragmentOrigin],
                attributes: [.font: statusLabel.font as Any],
                context: nil)
            let height = max(20, min(120, ceil(fitting.height) + 6))
            statusLabel.frame = NSRect(x: 40, y: (window.contentView?.bounds.height ?? 0) / 2 - height / 2,
                                       width: width - 80, height: height)
            statusLabel.autoresizingMask = [.width, .maxYMargin]
            statusLabel.cell?.wraps = true
            window.contentView?.addSubview(statusLabel)
            window.contentView?.subviews.last { $0 !== webView }?.isHidden = false
        } else {
            statusLabel.isHidden = true
        }
    }

    private func showBlockingError(title: String, message: String) {
        let alert = NSAlert()
        alert.alertStyle = .critical
        alert.messageText = title
        alert.informativeText = message
        alert.addButton(withTitle: "Open Log Folder")
        alert.addButton(withTitle: "Quit")
        NSApp.activate(ignoringOtherApps: true)
        if alert.runModal() == .alertFirstButtonReturn {
            NSWorkspace.shared.activateFileViewerSelecting([Paths.logFile])
        }
        NSApp.terminate(nil)
    }

    // MARK: navigation

    func webView(_ webView: WKWebView,
                 decidePolicyFor navigationAction: WKNavigationAction,
                 decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        guard let url = navigationAction.request.url else {
            decisionHandler(.cancel); return
        }
        // Anything not on our own backend is not ours to load.
        if url.host == "127.0.0.1" || url.host == "localhost" {
            decisionHandler(.allow)
        } else {
            NSWorkspace.shared.open(url)
            decisionHandler(.cancel)
        }
    }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        showBlockingError(title: "The interface failed to load", message: "\(error.localizedDescription)")
    }

    func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
        // The GPU process can be reclaimed under memory pressure. Reload rather
        // than leaving a blank window.
        webView.reload()
    }
}

// MARK: - entry

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.run()
