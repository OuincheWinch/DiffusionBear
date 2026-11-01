// Port selection for the local backend.
//
// Extracted from the app delegate so it can be exercised on its own. A copy in a
// test would be worthless here: the whole point of this file is that it decides
// whether a port belongs to us, and a re-implementation tests the re-implementation
// rather than the thing that ships. `swiftc -parse-as-library` plus a test main
// compiles THIS file, unmodified, against real sockets.
//
// WHY THIS EXISTS
// 8001 is a popular port. A dev server, an older build, or an unrelated app may
// hold it, and none of those should stop this app from starting. So the launcher
// walks a ladder: reuse a rung that already serves OUR backend, step over a rung
// held by anything else, and give up with a clear message only when the ladder is
// exhausted.

import Foundation

public enum PortProbe {
    /// Preferred port, and the rung of the ladder used when it is taken.
    public static let preferredPort = 8001
    public static let rung = 1024
    public static let maxAttempts = 10

    /// Every port that will be tried, in order.
    public static func ladder() -> [Int] {
        (0..<maxAttempts).map { preferredPort + $0 * rung }
    }

    public static func url(_ port: Int) -> URL {
        URL(string: "http://127.0.0.1:\(port)")!
    }

    /// Is anything listening on this TCP port?
    ///
    /// lsof is present on every macOS install; no added dependency. Its exit status
    /// is the signal, so stdout is discarded.
    public static func portInUse(_ port: Int) -> Bool {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/sbin/lsof")
        p.arguments = ["-nP", "-iTCP:\(port)", "-sTCP:LISTEN"]
        p.standardOutput = FileHandle.nullDevice
        p.standardError = FileHandle.nullDevice
        do {
            try p.run()
            p.waitUntilExit()
            return p.terminationStatus == 0
        } catch {
            // Cannot ask, so assume free rather than refusing to launch. The bind
            // will fail loudly if that guess was wrong, which is a better failure
            // than an app that never starts because lsof moved.
            return false
        }
    }

    /// Is the thing on this port OUR backend?
    ///
    /// "Any HTTP answer means uvicorn is up" is enough for a liveness check but not
    /// for deciding whether to REUSE a port. Something else on the machine may serve
    /// 8001 -- a dev server, an older build, an unrelated app -- and a 404 from it
    /// looks identical to a healthy answer under that looser test. So this asks for
    /// /api/version and requires JSON that names DiffusionBear AND carries a
    /// version-shaped field.
    ///
    /// Everything else answers false, which is the intended outcome: a timeout, an
    /// empty body, HTML, or JSON for some other service all mean "not ours, use
    /// another port".
    public static func answersAsOurs(_ port: Int, timeout: TimeInterval = 2.0) -> Bool {
        var request = URLRequest(url: url(port).appendingPathComponent("/api/version"))
        request.httpMethod = "GET"
        request.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        request.timeoutInterval = timeout

        let lock = NSLock()
        var payload: Data?
        var finished = false
        let done = DispatchSemaphore(value: 0)

        let task = URLSession.shared.dataTask(with: request) { data, response, _ in
            defer { done.signal() }
            guard let http = response as? HTTPURLResponse,
                  (200..<300).contains(http.statusCode),
                  let data = data
            else { return }
            lock.lock()
            payload = data
            finished = true
            lock.unlock()
        }
        task.resume()

        // Bounded on purpose. A port that accepts a connection and never completes
        // the request is exactly the case this must not hang on, so the wait is
        // capped rather than trusting the URLSession timeout alone.
        let result = done.wait(timeout: .now() + timeout + 0.25)
        if result != .success { task.cancel(); return false }

        lock.lock()
        defer { lock.unlock() }
        guard finished, let data = payload,
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let name = object["name"] as? String,
              let version = object["version"] as? String,
              !version.isEmpty
        else { return false }
        return name == "DiffusionBear"
    }

    /// Choose a port: reuse a rung already serving our backend, else take the first
    /// free one, stepping up the ladder. nil means the ladder is exhausted.
    public static func choosePort() -> Int? {
        for attempt in 0..<maxAttempts {
            let candidate = preferredPort + attempt * rung
            if answersAsOurs(candidate) { return candidate }
            if !portInUse(candidate) { return candidate }
        }
        return nil
    }
}