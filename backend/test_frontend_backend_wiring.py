"""Every API call the frontend makes must exist in the backend, with that method.

A button wired to a route that was renamed or removed fails silently: the click does
nothing and the UI reports nothing, which is exactly the "some buttons don't work"
report. Nothing caught it, because the frontend has no test runner and the backend
tests never look at the frontend.

So this reads the component sources, extracts every `api(...)` call and the method it
uses, and checks each against the live FastAPI route table. It is a static check, not a
live one: no request is made, so it is safe to run against a build with destructive
buttons in it, and it cannot delete anything.

The method matters as much as the path. `POST /api/models/refresh-licenses` exists;
`GET` on it does not. A path-only check would have called that button wired.
"""

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.routing import APIRoute

import main

SRC = Path(__file__).resolve().parents[1] / "frontend" / "src"

# Calls built from a variable or a template we cannot resolve statically. These are
# checked by hand rather than asserted here; listing them keeps the exclusion visible
# instead of silently dropping whatever fails to parse.
UNRESOLVABLE = {
    "/api/${source === \"civitai\" ? \"civitai\" : \"hf\"}/models",
}


def _canon(path: str) -> str:
    """Collapse parameter names so `/api/images/{id}` matches `{image_id}`."""
    path = re.sub(r"\$\{[^}]*\}", "{p}", path)
    return re.sub(r"\{[^}]+\}", "{p}", path)


def _walk(routes) -> list[APIRoute]:
    """Every APIRoute reachable from the app, including nested routers.

    This FastAPI version does not flatten `include_router` into `app.routes`; it keeps
    an `_IncludedRouter` wrapper holding `original_router`. Iterating `app.routes`
    directly therefore finds 2 routes instead of 73, and the first version of this test
    reported all forty frontend calls as broken -- including ones the running server
    answers with 200. A wiring test that cries wolf is worse than none, because the
    real failures go unread.
    """
    out: list[APIRoute] = []
    for route in routes:
        if isinstance(route, APIRoute):
            out.append(route)
        original = getattr(route, "original_router", None)
        if original is not None and hasattr(original, "routes"):
            out.extend(_walk(original.routes))
    return out


def _routes() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for route in _walk(main.app.routes):
        out.setdefault(_canon(route.path), set()).update(
            m for m in route.methods if m not in ("HEAD", "OPTIONS"))
    return out


def _call_sites() -> list[tuple[str, str, str]]:
    """(method, canonical path, source file) for every api() call in the frontend."""
    sites: list[tuple[str, str, str]] = []
    # api("literal", { ... method: "POST" ... })  -- the common shape.
    pattern = re.compile(
        r'api\(\s*[`"\']([^`"\']+)[`"\']\s*(?:,\s*\{(.*?)\}\s*)?\)',
        re.S,
    )
    for path in sorted(SRC.rglob("*.jsx")):
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            raw, options = match.group(1), match.group(2) or ""
            method = "GET"
            m = re.search(r'method:\s*"(\w+)"', options)
            if m:
                method = m.group(1).upper()
            sites.append((method, _canon(raw.split("?")[0]), path.name))
    return sites


class FrontendBackendIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.routes = _routes()
        cls.sites = _call_sites()

    def test_the_route_walker_finds_the_whole_app(self):
        """Guards the walker itself: 2 routes means nested routers were missed, and
        every assertion below would then report false failures."""
        self.assertGreater(len(self.routes), 50,
                           f"only {len(self.routes)} routes found; the walker is missing "
                           "nested routers")
        self.assertIn("/api/models", self.routes)
        self.assertIn("/api/settings", self.routes)

    def test_the_parser_actually_found_calls(self):
        """A parser that silently matches nothing makes every test below vacuous."""
        self.assertGreater(len(self.sites), 40,
                           "the api() parser stopped matching; these tests prove nothing")

    def test_every_frontend_call_has_a_backend_route(self):
        missing = {}
        for method, path, source in self.sites:
            if path in UNRESOLVABLE:
                continue
            if path not in self.routes:
                missing.setdefault(f"{method} {path}", set()).add(source)
        self.assertEqual(
            missing, {},
            "frontend calls with no backend route:\n"
            + "\n".join(f"    {p}  <- {sorted(s)}" for p, s in sorted(missing.items())),
        )

    def test_every_frontend_call_uses_an_allowed_method(self):
        wrong = {}
        for method, path, source in self.sites:
            if path in UNRESOLVABLE or path not in self.routes:
                continue
            if method not in self.routes[path]:
                wrong.setdefault(f"{method} {path}", set()).add(
                    f"{source} (backend has {sorted(self.routes[path])})")
        self.assertEqual(
            wrong, {},
            "frontend calls using a method the route does not accept:\n"
            + "\n".join(f"    {p}  <- {sorted(s)}" for p, s in sorted(wrong.items())),
        )

    def test_destructive_routes_still_require_confirmation(self):
        """The delete routes refuse a bare request.

        This is the guard that matters most here: these buttons are wired, they are
        supposed to work, and the only thing standing between a click and a user's
        model store is this check. Asserted so a future refactor cannot quietly drop
        it while "fixing" something else.
        """
        from fastapi.testclient import TestClient

        client = TestClient(main.app)
        r = client.request("DELETE", "/api/hf/models/does-not-exist-at-all")
        self.assertEqual(r.status_code, 422,
                         "a body-less DELETE must not be able to delete")

    def test_no_frontend_call_targets_a_path_outside_api(self):
        for _method, path, _source in self.sites:
            self.assertTrue(path.startswith("/api"), f"{path} is not an /api path")


if __name__ == "__main__":
    unittest.main()