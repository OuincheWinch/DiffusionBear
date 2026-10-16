"""Model routes live in routers.models, not routers.loras.

C7: model listing, inventory, licence refresh, uninstallation, local
installation and onboarding endpoints were embedded in routers/loras.py,
with model helpers alongside LoRA helpers. Downloads and conversion live in
routers/downloads.py, so model management had three homes and no owner.

This pins the ownership: every /api/models/* route must be registered on
routers.models.router, and the model helpers must be importable from
routers.models. Anything drifting back into loras.py fails here, not in
review.
"""
import unittest
from pathlib import Path

BACKEND = Path(__file__).resolve().parent


def _route_paths(router) -> set[str]:
    from fastapi.routing import APIRoute

    return {r.path for r in router.routes if isinstance(r, APIRoute)}


class TestModelRouteOwnership(unittest.TestCase):
    def test_models_router_owns_every_models_route(self):
        from routers import models as models_router

        paths = _route_paths(models_router.router)
        expected = {
            "/api/models",
            "/api/models/inventory",
            "/api/models/refresh-licenses",
            "/api/models/{model_id}",
            "/api/models/local-sources",
            "/api/models/install-local",
            "/api/models/onboard",
        }
        for path in expected:
            if path.startswith("/api/models"):
                with self.subTest(path=path):
                    self.assertIn(path, paths)

    def test_loras_router_has_no_models_routes(self):
        from routers import loras as loras_router

        for path in _route_paths(loras_router.router):
            self.assertFalse(
                path.startswith("/api/models"),
                f"{path} is a model route living on the loras router",
            )

    def test_model_helpers_live_in_models(self):
        from routers import models as models_router

        for name in (
            "_model_is_fully_cached",
            "_model_is_fully_cached_uncached",
            "_sort_models_for_picker",
            "_complete_sdxl_directory",
            "_complete_weight_directory",
            "_complete_hf_snapshot",
        ):
            with self.subTest(helper=name):
                self.assertTrue(
                    hasattr(models_router, name), f"routers.models.{name} missing")

    def test_app_serves_models_routes(self):
        import main

        def walk(routes):
            from fastapi.routing import APIRoute

            out = []
            for route in routes:
                if isinstance(route, APIRoute):
                    out.append(route)
                original = getattr(route, "original_router", None)
                if original is not None and hasattr(original, "routes"):
                    out.extend(walk(original.routes))
            return out

        paths = {r.path for r in walk(main.app.routes)}
        for path in (
            "/api/models", "/api/models/inventory", "/api/models/install-local",
            "/api/models/onboard", "/api/loras",
        ):
            with self.subTest(path=path):
                self.assertIn(path, paths)


if __name__ == "__main__":
    unittest.main()
