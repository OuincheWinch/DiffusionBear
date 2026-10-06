"""Fake listeners to test PortProbe's coherence check against."""
import json, socket, sys, threading, time

def serve(port, mode):
    s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", port)); s.listen(8)
    def handle(c):
        try:
            c.recv(65536)
            if mode == "blackhole":
                time.sleep(30); return
            if mode == "notfound":
                c.sendall(b"HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\nContent-Length: 9\r\n\r\nNot Found")
            elif mode == "html":
                b = b"<html>hello</html>"
                c.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: %d\r\n\r\n" % len(b) + b)
            elif mode == "otherjson":
                b = json.dumps({"name": "SomethingElse", "version": "9"}).encode()
                c.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(b) + b)
            elif mode == "jsonnoname":
                b = json.dumps({"foo": "bar"}).encode()
                c.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(b) + b)
        except Exception:
            pass
        finally:
            try: c.close()
            except Exception: pass
    def loop():
        while True:
            try: c, _ = s.accept()
            except OSError: return
            threading.Thread(target=handle, args=(c,), daemon=True).start()
    threading.Thread(target=loop, daemon=True).start()

if __name__ == "__main__":
    for spec in sys.argv[1:]:
        port, mode = spec.split(":")
        serve(int(port), mode)
    print("ready", flush=True)
    while True: time.sleep(3600)
