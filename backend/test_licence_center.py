"""The licence center has two tabs, and model licences are never invented.

Tab 1 (App & models) extends the existing LicencesTab with a models table
whose licence column only ever shows what a source stated -- verified today
against each repo's Hugging Face page. Tab 2 (Installed) lists installed
models and LoRAs with their licence blocks, falling back to a source link
(Civitai page, else HF repo) when no licence block exists.

Two things this pins down that regress silently:

1. A licence NAME in the models table must come from the verified set below.
   "other" is Hugging Face telling us the name lives elsewhere -- rendering it
   as a licence name repeats the exact violation LicenseBadges.jsx exists to
   prevent. If a vendor re-licences, this test fails and a human re-verifies;
   that friction is the point.
2. The installed tab must derive its fallback links from entry metadata
   (civitai_model_id / repo ids), never from hardcoded model-specific URLs,
   so custom-onboarded models and LoRAs -- which carry hf_repo/civitai_ids --
   get working links with no code change.
"""

import re
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "frontend" / "src"

# Licence names verified 2026-10-15 against each repo's own Hugging Face page,
# recorded with the repo URL beside each entry. Anything else -- including
# "other" -- must not appear as a licence name.
VERIFIED = {
    "apache-2.0",
    "creativeml-openrail-m",
    "openrail++",
    "cc-by-nc-nd-4.0",
}

# Ten built-in diffusion models plus the prompt-enhancer LLM. Compared
# case-insensitively: "Apache-2.0" as GitHub capitalises it is the same licence.
EXPECTED_MODEL_COUNT = 11

NEW_KEYS = [
    "licences.tabAppModels",
    "licences.tabInstalled",
    "licences.installedModels",
    "licences.installedLoras",
    "licences.installedEmpty",
]


def _read(relpath):
    return (SRC / relpath).read_text(encoding="utf-8")


def _strip_js_comments(text):
    """Remove JS comments without touching URL literals.

    A naive // stripper also eats the // in https:// inside template strings.
    This scanner skips single/double-quoted strings and template literals
    (with escapes), so reported line numbers stay right and URLs survive.
    """
    out = []
    i, n = 0, len(text)
    quote = None
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if quote:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(nxt)
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ("'", '"', "`"):
            quote = ch
            out.append(ch)
            i += 1
            continue
        if ch == "/" and nxt == "*":
            end = text.find("*/", i + 2)
            end = n if end < 0 else end + 2
            out.append("\n" * text.count("\n", i, end))
            i = end
            continue
        if ch == "/" and nxt == "/":
            end = text.find("\n", i)
            end = n if end < 0 else end
            i = end
            continue
        out.append(ch)
        i += 1
    return "".join(out)


class TestLicenceTabs(unittest.TestCase):
    def test_licences_tab_has_two_sub_tabs(self):
        code = _strip_js_comments(_read("components/LicencesTab.jsx"))
        self.assertRegex(code, r'useState\(\s*["\']app["\']\s*\)')
        for tab in ('"app"', '"installed"'):
            self.assertIn(tab, code, f"sub-tab {tab} missing from LicencesTab")

    def test_installed_tab_is_a_separate_component(self):
        # LicencesTab is already large; the installed list (two fetches, rows,
        # fallbacks) gets its own file instead of bloating it further.
        self.assertTrue(
            (SRC / "components" / "InstalledLicences.jsx").is_file(),
            "InstalledLicences.jsx missing",
        )

    def test_models_section_has_ten_verified_entries(self):
        data = _read("data/licences.js")
        names = re.findall(r"name:\s*\"([^\"]+)\"", data)
        # entries before the models section belong to dependency tables
        models_start = data.index("models")
        model_names = [
            m for m in re.finditer(r"name:\s*\"([^\"]+)\"", data)
            if m.start() > models_start
        ]
        self.assertEqual(
            len(model_names), EXPECTED_MODEL_COUNT,
            f"expected {EXPECTED_MODEL_COUNT} built-in models, found {len(model_names)}",
        )

    def test_no_invented_licence_names_in_models_section(self):
        data = _strip_js_comments(_read("data/licences.js"))
        models_start = data.index("models")
        shown = re.findall(r"license:\s*\"([^\"]+)\"", data[models_start:])
        for name in shown:
            self.assertIn(
                name.lower(), VERIFIED,
                f"licence {name!r} is not a verified source-stated value",
            )
        lowered = data[models_start:].lower()
        self.assertNotIn('"other"', lowered.replace("'", '"'))


class TestInstalledFallbacks(unittest.TestCase):
    def test_fetches_both_registries(self):
        code = _strip_js_comments(_read("components/InstalledLicences.jsx"))
        self.assertIn("/api/models", code)
        self.assertIn("/api/loras", code)

    def test_renders_badges_and_falls_back_to_source_links(self):
        code = _strip_js_comments(_read("components/InstalledLicences.jsx"))
        self.assertIn("LicenseBadges", code)
        self.assertIn("civitai.com/models/", code)
        self.assertIn("huggingface.co/", code)

    def test_no_hardcoded_model_urls_in_fallbacks(self):
        code = _strip_js_comments(_read("components/InstalledLicences.jsx"))
        for m in re.finditer(r"https://(civitai\.com|huggingface\.co)/\S+", code):
            self.assertRegex(
                m.group(0),
                r"\$\{|id|repo|ID|REPO",
                f"hardcoded model URL in fallback: {m.group(0)}",
            )


class TestLicenceCenterStrings(unittest.TestCase):
    def test_new_keys_exist_in_all_four_reference_languages(self):
        parts = _read("i18n/parts/licences.js")
        for key in NEW_KEYS:
            for lang in ("fr:", "en:", "de:", "it:"):
                block = re.search(
                    rf'"{re.escape(key)}":\s*\{{(.*?)\n  \}},', parts, flags=re.S)
                self.assertIsNotNone(block, f"{key} block missing")
                self.assertIn(lang, block.group(1), f"{key} missing {lang}")


if __name__ == "__main__":
    unittest.main()
