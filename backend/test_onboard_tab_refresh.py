"""Task 7: onboarding a custom model must appear in Generate with no restart.

Mirrors test_import_refreshes_registry.py: the declare -> review -> confirm flow
lives in a NEW OnboardModelTab (ImportTab is Draw Things-specific and stays
untouched), the register announces itself on a window event, and the Generate
picker re-reads /api/models when it hears it.

Static source checks: no frontend test runner exists in this repo.
"""

import re
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "frontend" / "src"
ONBOARD_TAB = SRC / "components" / "OnboardModelTab.jsx"
MODELS_TAB = SRC / "components" / "ModelsTab.jsx"
IMPORT_TAB = SRC / "components" / "ImportTab.jsx"
MODEL_CONFIG = SRC / "hooks" / "useModelConfig.js"
PARTS = SRC / "i18n" / "parts"

EVENT = "mlx:models-changed"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _callback_body(text: str, name: str) -> str:
    start = text.index(f"const {name} = useCallback(")
    depth, i = 0, text.index("{", start)
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
        i += 1
    raise AssertionError(f"unterminated callback {name}")


class TestOnboardComponent(unittest.TestCase):
    def test_component_exists_as_its_own_file(self):
        self.assertTrue(
            ONBOARD_TAB.exists(),
            "OnboardModelTab.jsx is missing; do not extend ImportTab.jsx instead",
        )

    def test_import_tab_stays_draw_things_specific(self):
        """ImportTab owns the Draw Things conversion flow; the onboard flow must
        not leak into it or a change here regresses a working tab."""
        self.assertNotIn("onboard", _read(IMPORT_TAB).lower())

    def test_confirm_posts_to_the_onboard_route(self):
        body = _callback_body(_read(ONBOARD_TAB), "confirm")
        self.assertIn('api("/api/models/onboard"', body)
        self.assertIn('method: "POST"', body)

    def test_hf_source_uses_the_hf_download_worker_with_declaration(self):
        """Local: nothing to fetch. HF id: the existing HF download worker, with
        the Task 5 family/variant declaration attached. The section slice pins
        both fields inside this leg's POST body, so a dropped variant fails."""
        body = _callback_body(_read(ONBOARD_TAB), "confirm")
        section = body[body.index('api("/api/models/download"'):body.index('api("/api/civitai/models/download"')]
        self.assertIn('method: "POST"', section)
        self.assertIn("family", section)
        self.assertIn("variant", section)

    def test_civitai_source_uses_the_civitai_download_worker_with_declaration(self):
        """Same pin for the Civitai leg: family without variant registers the
        wrong engine config just as silently."""
        body = _callback_body(_read(ONBOARD_TAB), "confirm")
        section = body[body.index('api("/api/civitai/models/download"'):]
        self.assertIn('method: "POST"', section)
        self.assertIn("family", section)
        self.assertIn("variant", section)

    def test_civitai_prefill_suggests_but_never_decides(self):
        """The version's baseModel tag may pre-select the family; the user's
        declaration still wins at confirm time."""
        self.assertIn("/api/civitai/models/version/", _read(ONBOARD_TAB))

    def test_review_step_precedes_confirm(self):
        """Declare -> review -> confirm, never auto-attach (spec section 2)."""
        text = _read(ONBOARD_TAB)
        self.assertRegex(text, r"review")
        body = _callback_body(text, "confirm")
        self.assertIn(EVENT, body)

    def test_event_fires_after_the_register_resolves(self):
        """Before the POST resolves the entry does not exist yet; a refresh
        issued now returns the OLD list."""
        body = _callback_body(_read(ONBOARD_TAB), "confirm")
        post = body.index('api("/api/models/onboard"')
        notify = body.index(EVENT)
        self.assertLess(post, notify, "registry notified before the entry is written")

    def test_a_window_event_is_dispatched(self):
        self.assertIn(EVENT, _read(ONBOARD_TAB))

    def test_status_poll_surfaces_repeated_failure(self):
        """The downloads poll is the only thing that can notice a stalled task;
        a catch that swallows forever wedges the spinner with no error box."""
        text = _read(ONBOARD_TAB)
        self.assertIn('api("/api/models/downloads")', text)
        self.assertRegex(text, r"MAX_POLL_FAILURES\s*=\s*\d+")
        poll = text[text.index('api("/api/models/downloads")'):]
        catch = poll.index("} catch")
        handler = poll[catch:catch + 600]
        self.assertIn("setError", handler)

    def test_no_bare_failure(self):
        """Every refusal names its reason through the error box, never a bare
        failure (spec section 4)."""
        text = _read(ONBOARD_TAB)
        self.assertIn("setError", text)
        for m in re.finditer(r"catch\s*(?:\(\s*\w+\s*\))?\s*\{([^{}]*)\}", text):
            self.assertTrue(
                m.group(1).strip(),
                "empty catch swallows the refusal reason",
            )


class TestCivitaiRefShapes(unittest.TestCase):
    """The reported failure: a civitai.red model-page URL and Civitai download
    links never parsed, so both died in the same red banner. Only numbers are
    ever extracted -- the URL itself is never fetched, so mirrors cannot
    smuggle a destination in; the backend resolves everything by id."""

    def test_mirror_domain_parses(self):
        text = _read(ONBOARD_TAB)
        self.assertRegex(text, r"civitai\\\.\(com\|red\)|civitai\\\.\(\?:com\|red\)")

    def test_download_link_shape_parses_to_version_only(self):
        text = _read(ONBOARD_TAB)
        self.assertIn("api/download/models/", text)

    def test_version_only_resolves_model_id_before_download(self):
        """A pasted ref without a model id must be resolved through prefill
        before the download POST, which requires both ids. Resolving after
        (or never) reproduces the reported banner."""
        body = _callback_body(_read(ONBOARD_TAB), "confirm")
        prefill_at = body.index("/api/civitai/models/version/")
        download_at = body.index('api("/api/civitai/models/download"')
        self.assertLess(
            prefill_at, download_at,
            "download attempted before version-only input resolved its model id",
        )


class TestPickerListens(unittest.TestCase):
    def test_model_config_subscribes_to_the_event(self):
        text = _read(MODEL_CONFIG)
        self.assertIn(EVENT, text)
        self.assertRegex(text, rf'addEventListener\(\s*"{EVENT}"')

    def test_listener_refreshes_the_model_list(self):
        """Subscribing and ignoring would pass a listener-exists check while the
        picker stays stale until restart -- the original LoRA bug, repeated."""
        text = _read(MODEL_CONFIG)
        self.assertRegex(
            text,
            r'const onModelsChanged = \(\) => \{\s*loadModels\(\);\s*\};'
            rf'[\s\S]{{0,200}}?addEventListener\(\s*"{EVENT}",\s*onModelsChanged\)',
        )

    def test_listener_is_removed_on_unmount(self):
        text = _read(MODEL_CONFIG)
        self.assertRegex(
            text, rf'addEventListener\(\s*"{EVENT}"[\s\S]{{0,600}}?removeEventListener\(\s*"{EVENT}"'
        )


class TestWiring(unittest.TestCase):
    def test_models_tab_renders_the_onboard_flow_on_the_import_side(self):
        text = _read(MODELS_TAB)
        self.assertRegex(text, r"<OnboardModelTab[^>]*")
        self.assertRegex(text, r'import OnboardModelTab from "\./OnboardModelTab"')

    def test_onboard_strings_are_all_in_the_catalogue(self):
        """Every t() key the component uses must exist in parts/ (fr/en/de/it);
        test_i18n.py then holds the nine-language line."""
        keys = set(re.findall(r'\bt\(\s*"([^"]+)"', _read(ONBOARD_TAB)))
        self.assertTrue(keys, "no t() keys found; strings must go through the catalogue")
        catalog = set()
        for part in PARTS.glob("*.js"):
            catalog.update(re.findall(r'^\s{2,}"([^"]+)":\s*\{', part.read_text(encoding="utf-8"), flags=re.M))
        missing = sorted(k for k in keys if k not in catalog)
        self.assertEqual(missing, [], f"component keys missing from the catalogue: {missing}")


if __name__ == "__main__":
    unittest.main()
