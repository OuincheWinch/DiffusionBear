import sys
import unittest
from pathlib import Path
from starlette.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

import main


class MiddlewareSecurityTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)

    def test_localhost_middleware_allows_loopback_client(self):
        # testclient client host defaults to "testclient", which _is_loopback accepts
        res = self.client.get("/api/settings", headers={"Host": "localhost:8001"})
        # Should pass LocalHostMiddleware (not 400 or 403 from localhost check)
        self.assertNotEqual(res.status_code, 400)
        self.assertNotEqual(res.status_code, 403)

    def test_localhost_middleware_rejects_disallowed_host_header(self):
        res = self.client.get("/api/settings", headers={"Host": "evil.com:8001"})
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json(), {"detail": "host is not allowed"})

    def test_localhost_middleware_rejects_non_loopback_client_ip(self):
        # Even if Host: localhost is provided, non-loopback client IP is refused (M1)
        external_client = TestClient(main.app, client=("192.168.1.100", 54321))
        # Non-API path protected by LocalHostMiddleware
        res_non_api = external_client.get(
            "/",
            headers={"Host": "localhost:8001"},
        )
        self.assertEqual(res_non_api.status_code, 403)
        self.assertEqual(res_non_api.json(), {"detail": "access is restricted to loopback clients"})

        # API path protected by LocalAPIMiddleware
        res_api = external_client.get(
            "/api/settings",
            headers={"Host": "localhost:8001"},
        )
        self.assertEqual(res_api.status_code, 403)
        self.assertIn("restricted to loopback clients", res_api.json()["detail"])

    def test_request_size_middleware_rejects_large_content_length(self):
        res = self.client.post(
            "/api/settings",
            headers={"Content-Length": str(10 * 1024 * 1024)},
            content=b"",
        )
        self.assertEqual(res.status_code, 413)
        self.assertEqual(res.json(), {"detail": "request body is too large"})

    def test_request_size_middleware_rejects_large_chunked_stream(self):
        # Streaming chunked body totaling > 8MB (M8)
        chunk = b"x" * (1024 * 1024)  # 1MB chunk

        def stream_large():
            for _ in range(9):  # 9MB total
                yield chunk

        res = self.client.post(
            "/api/settings",
            content=stream_large(),
        )
        self.assertEqual(res.status_code, 413)
        self.assertEqual(res.json(), {"detail": "request body is too large"})


if __name__ == "__main__":
    unittest.main()
