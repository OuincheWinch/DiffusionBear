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
private let startupTimeout: TimeInterval = 300
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

// MARK: - Backend supervision

/// Owns the backend child process: starts it, waits for it to answer, watches it,
/// and makes sure it is gone when the app quits.
final class BackendProcess {
    private var process: Process?
    private var logHandle: FileHandle?
    private var pollTimer: Timer?
    private(set) var isReady = false
    var onReady: (() -> Void)?
    var onFailure: ((String) -> Void)?

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

        // Truncate the log so a long-running app does not grow it forever, and so
        // "Reveal in Finder" always shows this session.
        if !FileManager.default.fileExists(atPath: Paths.logFile.path) {
            FileManager.default.createFile(atPath: Paths.logFile.path, contents: nil)
        } else {
            try? Data().write(to: Paths.logFile)
        }
        logHandle = try? FileHandle(forWritingTo: Paths.logFile)

        let p = Process()
        p.executableURL = Paths.python
        p.arguments = ["-m", "uvicorn", "main:app",
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
        NSLog("DiffusionBear: backend pid %d, cwd %@", p.processIdentifier, Paths.backend.path)

        let started = Date()
        pollTimer = Timer.scheduledTimer(withTimeInterval: pollInterval, repeats: true) { [weak self] timer in
            guard let self else { timer.invalidate(); return }
            if p.isRunning == false {
                timer.invalidate()
                self.onFailure?("The backend exited during startup.\n\nLog:\n\(Paths.logFile.path)")
                return
            }
            if Self.answers(path: "/api/version") {
                timer.invalidate()
                self.isReady = true
                self.onReady?()
                return
            }
            if Date().timeIntervalSince(started) > startupTimeout {
                timer.invalidate()
                self.onFailure?("The backend did not become ready within \(Int(startupTimeout))s.\n\nLog:\n\(Paths.logFile.path)")
            }
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

// MARK: - UI

final class AppDelegate: NSObject, NSApplicationDelegate, WKNavigationDelegate {
    private var window: NSWindow!
    private var webView: WKWebView!
    private let backend = BackendProcess()
    private var statusLabel: NSTextField!
    private var activityToken: NSObjectProtocol?

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
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
        webView = WKWebView(frame: NSRect(x: 0, y: 0, width: 1280, height: 840), configuration: config)
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
            styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
            backing: .buffered, defer: false)
        window.title = "DiffusionBear"
        window.titlebarAppearsTransparent = true
        window.titleVisibility = .hidden
        window.isReleasedWhenClosed = false
        window.minSize = NSSize(width: 900, height: 600)
        window.contentView = content
        window.center()
        window.makeKeyAndOrderFront(nil)
        window.makeFirstResponder(webView)
    }

    private func setStatus(_ text: String?, spinner: Bool = false) {
        guard let window else { return }
        if let text {
            statusLabel.stringValue = spinner ? "\(text)" : text
            statusLabel.isHidden = false
            // Float the status over the (still blank) web view.
            statusLabel.frame = NSRect(x: 0, y: (window.contentView?.bounds.height ?? 0) / 2 - 10,
                                       width: window.contentView?.bounds.width ?? 0, height: 20)
            statusLabel.autoresizingMask = [.width, .maxYMargin]
            window.contentView?.addSubview(statusLabel)
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
