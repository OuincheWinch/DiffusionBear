"""Importing a LoRA must make it appear immediately, not after a restart.

Reported on 2026-10-13: convert a Draw Things LoRA, press "Confirm import", the app
jumps to the Generate tab, and the new LoRA is not in the installed list. It turns up
after a reboot.

Nothing was broken. Both writes succeeded:

  * the file went to ASSET_DIR/lora_files, and
  * the registry entry went to DATA_DIR/loras.json,

and `/api/loras` returned the new entry the whole time. The fault is that no client
ever asked again. After the POST, `ImportTab` called its own `load()`, which re-reads
`/api/sources/drawthings/files` -- the list of `.ckpt` files still sitting in the Draw
Things folder, which the import did not change. The registry it had just written was
never re-requested.

A reboot looked like the fix because every tab is kept mounted (this app has no router;
sections are hidden with display:none), so the registry is fetched exactly once, on
mount. Restart is the only thing that re-mounts it.

The hard part is that the two halves are in different components and there is no shared
store: `ImportTab` lives in the Models tab, the registry lives in `useLoraPanel` inside
GenerateForm. So the notification has to cross that gap, and this file pins that it
does -- including the ORDER, because calling back before the POST resolves announces an
import that has not happened yet.

Static source checks, as with test_frontend_no_silent_failures.py: no frontend test
runner exists in this repo, so these assert structure rather than behaviour.
"""
import re
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "frontend" / "src"
IMPORT_TAB = SRC / "components" / "ImportTab.jsx"
MODELS_TAB = SRC / "components" / "ModelsTab.jsx"
LORA_PANEL = SRC / "hooks" / "useLoraPanel.js"

IMPORT_TEXT = IMPORT_TAB.read_text(encoding="utf-8")
MODELS_TEXT = MODELS_TAB.read_text(encoding="utf-8")
PANEL_TEXT = LORA_PANEL.read_text(encoding="utf-8")

EVENT = "mlx:loras-changed"


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


class TestImportNotifiesTheRegistry(unittest.TestCase):
    def test_import_tab_accepts_a_callback(self):
        """With no prop, the component has no channel to the rest of the app at all."""
        self.assertRegex(
            IMPORT_TEXT,
            r"export default function ImportTab\(\s*\{[^}]*onImported",
            "ImportTab takes no props, so it cannot tell the app a LoRA was added",
        )

    def test_callback_fires_after_the_post_resolves(self):
        """Before the POST resolves, the registry entry does not exist yet, and a
        refresh issued now returns the OLD list -- the import still looks like it
        did nothing."""
        body = _callback_body(IMPORT_TEXT, "confirm")
        post = body.index('api("/api/loras", { method: "POST"')
        notify = body.index(EVENT)
        self.assertLess(post, notify,
                        "the registry is notified before the import is written")

    def test_a_window_event_is_dispatched(self):
        """The registry lives in another component in another tab, with no shared
        store. The app already has this convention (`mlx:load-prompt`,
        `mlx:cancel-job`), so the notification follows it rather than inventing one."""
        self.assertIn(EVENT, IMPORT_TEXT)

    def test_existing_load_call_does_not_count_as_a_refresh(self):
        """Regression guard for the original bug, which LOOKED handled: `load()`
        exists right where a refresh should be, and refreshes the wrong endpoint."""
        body = _callback_body(IMPORT_TEXT, "confirm")
        self.assertIn("await load()", body)
        # load() must stay pointed at the Draw Things listing.
        load = _callback_body(IMPORT_TEXT, "load")
        self.assertIn("/api/sources/drawthings/files", load)
        self.assertNotIn("/api/loras", load)


class TestRegistryListens(unittest.TestCase):
    def test_lora_panel_subscribes_to_the_event(self):
        self.assertIn(EVENT, PANEL_TEXT)
        self.assertRegex(PANEL_TEXT, rf'addEventListener\(\s*"{EVENT}"')

    def test_listener_refreshes_the_registry(self):
        """Subscribing and ignoring would look identical to a typo in the event
        name, and would pass a test that only checks the listener exists.

        The handler is defined immediately before it is registered, so the refresh is
        asserted on the handler itself rather than on text following addEventListener
        -- asserting on position instead of on the call is how the first version of
        this test failed on correct code."""
        self.assertRegex(
            PANEL_TEXT,
            r'const onLorasChanged = \(\) => \{\s*refreshLoraRegistry\(\);\s*\};'
            r'[\s\S]{0,200}?addEventListener\(\s*"mlx:loras-changed",\s*onLorasChanged\)',
        )

    def test_listener_is_removed_on_unmount(self):
        """An effect with no cleanup leaves a listener on a torn-down component, which
        keeps the closure (and its state setters) alive for the rest of the session."""
        self.assertRegex(PANEL_TEXT, rf'addEventListener\(\s*"{EVENT}"[\s\S]{{0,600}}?removeEventListener\(\s*"{EVENT}"')


class TestWiring(unittest.TestCase):
    def test_models_tab_passes_the_callback(self):
        """ModelsTab owns the model-registry refresh already (SourceBrowser gets it as
        onInstalled). ImportTab has to be given the same door."""
        self.assertRegex(MODELS_TEXT, r"<ImportTab[^>]*onImported")


if __name__ == "__main__":
    unittest.main()