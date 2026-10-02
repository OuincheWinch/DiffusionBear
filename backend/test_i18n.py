"""Guards the translation table.

A missing or empty language used to be invisible: the UI renders whatever is in the
table, so a key that exists for French but not German shows an empty label in German.
This fails the build instead.

The table is parsed with a regex rather than a JS parser on purpose -- there is no
node-based test runner in this repo, and the file's shape (a flat object of string
literals) is stable enough to check without executing JavaScript.
"""

import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

STRINGS_FILE = Path(__file__).resolve().parents[1] / "frontend" / "src" / "i18n" / "strings.js"
LANGUAGES_FILE = Path(__file__).resolve().parents[1] / "frontend" / "src" / "i18n" / "languages.js"
LANGS = ("fr", "en", "de", "it")

# Both layouts have to parse. The catalogues are hand-maintained and prettier has
# not normalised them, so roughly half the entries sit on one line
# (`  "key": { fr: "...", en: "..." },`) and half span several. The first version of
# this parser only handled the multi-line form and silently "verified" 449 of 674
# keys, which is worse than no check at all.
_KEY_RE = re.compile(r'^\s{2}"(?P<key>[^"]+)":\s*\{', re.M)
# The trailing lookahead accepts end-of-body as well as a comma or brace: entry
# bodies are sliced to exclude the closing '}', so the LAST language of every entry
# ends the string. Without `$` that language was never matched, and the test failed
# on exactly one gap per key -- which is how 'it' went missing across the catalogue.
_LANG_RE = re.compile(
    r'(?<![A-Za-z0-9_])(?P<lang>fr|en|de|it)\s*:\s*"', re.M
)

# String literals are blanked before the language keys are looked for, so a
# translation that happens to contain the text `de: "..."` cannot be mistaken for a
# real entry. It also removes the need for separators to be positioned exactly,
# which is what made the previous pattern miss `fr`/`en`/`it` on single-line
# entries and silently under-count the catalogue.
_STR_RE = re.compile(r'"(?:[^"\\]|\\.)*"')


def _strip_strings(body: str) -> str:
    """Blank the contents of every string literal, keeping the quotes.

    Offsets and length are preserved, so a span found in this skeleton maps to the
    same span in the original text. The quote characters stay because the language
    matcher keys on them: blanking them outright made the pattern match nothing at
    all, and replacing them with a shorter marker shifted every span.
    """
    return _STR_RE.sub(lambda m: '"' + " " * (len(m.group(0)) - 2) + '"', body)


def _split_entries(text: str) -> list:
    """Yield (key, body) for each top-level entry, honouring nesting.

    A regex alone is not enough: a value can contain a brace (the French "Effacer ET
    oublier ?" confirm text embeds a sentence, and licence names may too), so the
    body of an entry runs to its matching close brace, not to the next key at the
    same indent. The earlier line-based version mis-sliced exactly that entry and
    reported a placeholder mismatch that did not exist.
    """
    out = []
    for m in _KEY_RE.finditer(text):
        depth = 0
        i = m.end() - 1          # the '{' that _KEY_RE stopped on
        start = i + 1
        in_str = False
        esc = False
        while i < len(text):
            c = text[i]
            if in_str:
                # Braces inside a string literal are data, not structure. The
                # "{ } JSON" button label is the case that bit this: counting its
                # braces merged that entry with the next one, so the following key
                # inherited two languages and lost the other two.
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    out.append((m.group("key"), text[start:i]))
                    break
            i += 1
    return out


def _parse_table(path: Path | None = None) -> dict:
    """Return {key: {lang: raw literal}} for strings.js or one parts/*.js file."""
    text = (path or STRINGS_FILE).read_text(encoding="utf-8")
    entries = {}
    for key, body in _split_entries(text):
        # Blank the literals for structure detection, but keep the originals for
        # the values, so the two are matched up by language code.
        skeleton = _strip_strings(body)
        langs = {}
        for m in _LANG_RE.finditer(skeleton):
            # The skeleton has the same offsets as the body, so the language code
            # and its literal can be sliced straight out of the original text.
            lit = _STR_RE.search(body, m.end() - 1)
            if lit is not None:
                langs[m.group("lang")] = lit.group(0)
        entries[key] = langs
    return entries


def _unquote(raw: str) -> str:
    body = raw[1:-1]
    return (
        body.replace("\\n", "\n")
        .replace('\\"', '"')
        .replace("\\\\", "\\")
    )


PART_FILES = sorted((Path(__file__).resolve().parents[1] / "frontend" / "src" / "i18n" / "parts").glob("*.js"))


def _parse_parts() -> dict:
    """Parse every ./parts/*.js catalogue the same way as strings.js."""
    entries = {}
    for path in PART_FILES:
        for key, langs in _parse_table(path).items():
            if key in entries:
                raise AssertionError(f"duplicate key across part files: {key}")
            entries[key] = langs
    return entries


class TranslationTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.entries = _parse_table()
        cls.parts = _parse_parts()

    def test_table_parses(self):
        # The core table shrank on purpose: most strings moved to parts/*.js, which
        # are the real home now. This only guards against the parser silently
        # returning nothing, so the bound is low and test_part_files_parse carries
        # the meaningful volume check.
        self.assertGreater(
            len(self.entries), 20, "the core translation table looks unparsed"
        )

    def test_part_files_parse(self):
        self.assertGreater(
            len(self.parts), 500, f"only {len(self.parts)} keys parsed from parts/*.js"
        )
        self.assertTrue(PART_FILES, "no catalogue part files found at all")

    def test_no_placeholder_part_files(self):
        """An unwritten part file would silently ship a component in one language."""
        empty = [p.name for p in PART_FILES if not _parse_table(p)]
        self.assertEqual(
            empty, [], f"part files with no keys (untranslated?): {empty}"
        )

    def test_every_part_key_has_every_language(self):
        missing = {}
        for key, langs in self.parts.items():
            gaps = [lang for lang in LANGS if lang not in langs]
            if gaps:
                missing[key] = gaps
        self.assertEqual(missing, {}, f"part keys missing languages: {missing}")

    def test_no_blank_part_translation(self):
        blanks = {
            key: lang
            for key, langs in self.parts.items()
            for lang, raw in langs.items()
            if not _unquote(raw).strip()
        }
        self.assertEqual(blanks, {}, f"blank part translations: {blanks}")

    def test_part_keys_are_namespaced(self):
        bad = [k for k in self.parts if "." not in k]
        self.assertEqual(bad, [], f"un-namespaced part keys: {bad}")

    def test_part_placeholders_match_across_languages(self):
        mismatched = {}
        for key, langs in self.parts.items():
            names = {
                lang: set(re.findall(r"\{(\w+)\}", _unquote(raw)))
                for lang, raw in langs.items()
            }
            reference = names.get("en") or set()
            for lang, found in names.items():
                if found != reference:
                    mismatched[key] = (lang, sorted(found), sorted(reference))
        self.assertEqual(
            mismatched,
            {},
            f"part placeholder mismatch (lang, found, expected): {mismatched}",
        )

    def test_no_key_collides_between_core_and_parts(self):
        collisions = sorted(set(self.entries) & set(self.parts))
        self.assertEqual(collisions, [], f"keys defined twice: {collisions}")

    def test_every_key_has_every_language(self):
        missing = {}
        for key, langs in self.entries.items():
            gaps = [lang for lang in LANGS if lang not in langs]
            if gaps:
                missing[key] = gaps
        self.assertEqual(missing, {}, f"keys missing languages: {missing}")

    def test_no_language_is_blank(self):
        blanks = {
            key: lang
            for key, langs in self.entries.items()
            for lang, raw in langs.items()
            if not _unquote(raw).strip()
        }
        self.assertEqual(blanks, {}, f"blank translations: {blanks}")

    def test_keys_are_namespaced(self):
        bad = [k for k in self.entries if "." not in k]
        self.assertEqual(bad, [], f"un-namespaced keys: {bad}")

    def test_placeholders_match_across_languages(self):
        """A translation must not drop or invent a {placeholder}."""
        mismatched = {}
        for key, langs in self.entries.items():
            names = {
                lang: set(re.findall(r"\{(\w+)\}", _unquote(raw)))
                for lang, raw in langs.items()
            }
            reference = names.get("en") or set()
            for lang, found in names.items():
                if found != reference:
                    mismatched[key] = (lang, sorted(found), sorted(reference))
        self.assertEqual(
            mismatched, {}, f"placeholder mismatch (lang, found, expected): {mismatched}"
        )

    def test_languages_module_lists_the_same_codes(self):
        text = LANGUAGES_FILE.read_text(encoding="utf-8")
        for code in ("en", "fr", "de", "it"):
            self.assertIn(f'code: "{code}"', text, f"{code} missing from languages.js")


class NoHardcodedFrenchTests(unittest.TestCase):
    """The app shell is not French-only any more.

    This is a blunt check by necessity: it flags French text sitting in JSX bodies
    of the components that were translated. It only looks at files that already
    import useI18n, so untouched components are not flagged yet.
    """

    # Markers that only exist in French. Deliberately excludes "prompts" and similar
    # cognates: they are also correct English and German, so matching them produced
    # false positives on already-translated code (t("queue.promptsCount") contains it).
    FRENCH_MARKERS = (
        "réinsér", "Récupérable", "récupérable", "effacer", "Effacer",
        "Chargement", "Générer", "Paramètres", "Annuler", "Fermer",
        "Supprimer", "Enregistrer", "Interrompu", "Annulé",
        "modèle standard", "Actualiser", "Réessayer",
    )

    def test_translated_components_have_no_french_left(self):
        offenders = {}
        for path in Path(__file__).resolve().parents[1].joinpath("frontend/src/components").rglob("*.jsx"):
            text = path.read_text(encoding="utf-8")
            if "useI18n" not in text:
                continue
            # A key name in a t(...) call is not UI text, so drop call arguments
            # before scanning; only literal JSX and string values are of interest.
            body = re.sub(r't\(\s*"[^"]*"', 't("__KEY__"', text)
            found = sorted({m for m in self.FRENCH_MARKERS if m in body})
            if found:
                offenders[str(path.name)] = found
        self.assertEqual(offenders, {}, f"untranslated French left in: {offenders}")


if __name__ == "__main__":
    unittest.main()


# --------------------------------------------------------------------------- scaffolds
#
# Five languages (es, zh, ja, pt, ko) ship as empty catalogues: selectable in the
# picker, rendering through the English fallback until translated. The scaffolds are
# the translators' work order, and these tests keep them honest without pretending
# the translations exist.

ROOT = Path(__file__).resolve().parents[1]
SCAFFOLD_DIR = ROOT / "frontend" / "src" / "i18n" / "scaffold"
SETTINGS_ROUTER = Path(__file__).resolve().parent / "routers" / "settings.py"
APP_SETTINGS = Path(__file__).resolve().parent / "app_settings.py"

# The codes that must have a scaffold file, hard-coded here on purpose. Deriving the
# expectation from languages.js would make these tests agree with any list it holds,
# including one that quietly forgot a language.
EXPECTED_SHIPPED = ("en", "fr", "de", "it", "es", "zh", "ja", "pt", "ko")
EXPECTED_SCAFFOLD = ()
# Declared untranslated even though these five were translated. Each is a brand name or a
# closed-circuit technical literal, so "DiffusionBear" is the correct Spanish value.
_INTENTIONALLY_UNTRANSLATED = {"app.logoAlt"}

_KEY_LINE_RE = re.compile(r'^\s{2}"(?P<key>[^"]+)":\s*(?P<value>.+?),?\s*(?://.*)?$', re.M)
_CODE_RE = re.compile(r'code:\s*"(?P<code>[a-z]{2})"')
# Anchored on the whole declaration, because settings.py holds several `pattern=` fields
# and `language: str | None = Field(` contains an `=`, so the shorter `language:[^=]*`
# form stops before reaching the pattern.
_LANG_FIELD_RE = re.compile(
    r'language:\s*str\s*\|\s*None\s*=\s*Field\(\s*default=None,\s*'
    r'pattern=r"\^\((?P<codes>[^)]*)\)\$"',
    re.S,
)


def _registry_languages() -> list[dict]:
    """The LANGUAGES array from languages.js, parsed without executing it."""
    text = LANGUAGES_FILE.read_text(encoding="utf-8")
    block = re.search(r"export const LANGUAGES = \[(.*?)\n\];", text, re.S).group(1)
    out = []
    for raw in re.findall(r"\{[^{}]*\}", block):
        def field(name):
            m = re.search(rf'{name}:\s*"([^"]+)"', raw)
            return m.group(1) if m else None
        out.append(
            {
                "code": field("code"),
                "label": field("label"),
                "flag": field("flag"),
                "status": field("status"),
            }
        )
    return out


class LanguageRegistryTests(unittest.TestCase):
    def setUp(self):
        self.languages = _registry_languages()

    def test_exactly_nine_languages_are_listed(self):
        self.assertEqual(
            [lang["code"] for lang in self.languages],
            list(EXPECTED_SHIPPED) + list(EXPECTED_SCAFFOLD),
        )

    def test_codes_are_unique(self):
        codes = [lang["code"] for lang in self.languages]
        self.assertEqual(len(codes), len(set(codes)))

    def test_every_language_has_a_native_name_and_a_flag(self):
        for lang in self.languages:
            with self.subTest(code=lang["code"]):
                self.assertTrue(lang["label"], "native name missing")
                self.assertTrue(lang["flag"], "flag missing")
                # These are the labels the user recognises, in their own script.
                self.assertNotIn("language.", lang["label"])

    def test_native_names_are_the_ones_asked_for(self):
        expected = {
            "es": "Español",
            "zh": "简体中文",
            "ja": "日本語",
            "pt": "Português",
            "ko": "한국어",
        }
        got = {lang["code"]: lang["label"] for lang in self.languages}
        for code, label in expected.items():
            with self.subTest(code=code):
                self.assertEqual(got[code], label)

    def test_flags_match_the_language(self):
        expected = {"en": "🇬🇧", "fr": "🇫🇷", "de": "🇩🇪", "it": "🇮🇹",
                    "es": "🇪🇸", "zh": "🇨🇳", "ja": "🇯🇵", "pt": "🇵🇹", "ko": "🇰🇷"}
        for lang in self.languages:
            with self.subTest(code=lang["code"]):
                self.assertEqual(lang["flag"], expected[lang["code"]])

    def test_status_is_declared_and_partitions_the_list(self):
        statuses = {lang["code"]: lang["status"] for lang in self.languages}
        self.assertEqual({c for c, s in statuses.items() if s == "shipped"}, set(EXPECTED_SHIPPED))
        self.assertEqual({c for c, s in statuses.items() if s == "scaffold"}, set(EXPECTED_SCAFFOLD))

    def test_an_untranslated_language_is_not_auto_selected(self):
        """A locale with no catalogue must fall back to English, or the picker claims
        Español while the interface is English and reads as broken."""
        text = LANGUAGES_FILE.read_text(encoding="utf-8")
        self.assertIn("TRANSLATED_LANGUAGES", text)
        self.assertRegex(
            text,
            r"TRANSLATED_LANGUAGES\.some\(\(l\) => l\.code === fromNav\)",
            "detectLanguage must check the OS locale against the translated set",
        )


class BackendLanguageAgreementTests(unittest.TestCase):
    """A language the picker offers but the backend rejects is a dead control: the
    choice appears to do nothing. Both gates are asserted against the registry."""

    def _registry_codes(self) -> set:
        return {lang["code"] for lang in _registry_languages()} | {"auto"}

    def test_the_settings_router_accepts_exactly_the_registry(self):
        router = SETTINGS_ROUTER.read_text(encoding="utf-8")
        match = re.search(_LANG_FIELD_RE, router)
        self.assertIsNotNone(match, "language pattern not found in routers/settings.py")
        accepted = set(match.group("codes").split("|"))
        self.assertEqual(accepted, self._registry_codes())

    def test_app_settings_accepts_exactly_the_registry(self):
        text = APP_SETTINGS.read_text(encoding="utf-8")
        match = re.search(r'LANGUAGE_CODES = frozenset\(\{(?P<codes>[^}]*)\}\)', text)
        self.assertIsNotNone(match, "LANGUAGE_CODES not found in app_settings.py")
        accepted = set(re.findall(r'"([a-z]{2}|auto)"', match.group("codes")))
        self.assertEqual(accepted, self._registry_codes())

    def test_the_validator_actually_uses_the_shared_constant(self):
        self.assertRegex(
            APP_SETTINGS.read_text(encoding="utf-8"),
            r'"language":\s*lambda v:\s*v in LANGUAGE_CODES',
            "a literal tuple here would drift from the constant above it",
        )

    def test_all_nine_codes_are_persistable(self):
        router = SETTINGS_ROUTER.read_text(encoding="utf-8")
        # Anchored to `language:` on purpose -- settings.py holds several `pattern=`
        # fields, and an unanchored match happily reads image formats instead.
        match = re.search(_LANG_FIELD_RE, router)
        self.assertIsNotNone(match)
        accepted = match.group("codes").split("|")
        for code in ("es", "zh", "ja", "pt", "ko"):
            with self.subTest(code=code):
                self.assertIn(code, accepted)


_OVERLAY_CODES = ("es", "zh", "ja", "pt", "ko")
_SCRIPT_RANGES = {
    "han": r"[\u4e00-\u9fff\u3400-\u4dbf]",
    "kana": r"[\u3040-\u30ff]",
    "hangul": r"[\uac00-\ud7af\u1100-\u11ff]",
}
# Latin words that legitimately appear inside CJK text: product and vendor names, licence
# identifiers, algorithms, file formats. Anything else in a CJK value is corruption -- a
# fragment of another language that leaked in while authoring five languages in one pass.
_ALLOWED_LATIN = {
    "Civitai", "DeepCache", "FLUX", "LoRA", "LoRAs", "MIT", "SPA", "React", "Vite",
    "Python", "torch", "SDXL", "sdxl", "Krea", "KREA", "Hugging", "Face", "Z-Image",
    "Qwen", "Image", "klein", "Turbo", "Juggernaut", "XL", "Lightning", "DiffusionBear",
    "CoreML", "Apple", "safetensors", "venv", "UNet", "VAE",
    # Abbreviations and algorithms that read as-is in all five languages.
    "OOM", "HF", "ID", "URL", "AI", "UI", "API", "MP", "px", "In", "Context",
    "Copyright", "MLX", "mflux", "transformer", "token",
    "Metal", "CFG", "SOTA", "Lanczos", "PNG", "JPEG", "JPG", "WebP", "Cmd", "NSFW",
    "AppKit", "WebKit", "macOS", "Finder", "Civit", "GPU", "Silicon", "JSON", "Web",
    "EXIF", "DIFFUSION", "Fast", "LLM", "Instruct", "Klein", "TAEF", "TAESD",
    "krea", "mlx", "qwen", "blob", "snapshot", "snapshots", "TASK",
}


def _latin_words(text: str) -> set:
    """Latin words in a value, ignoring the parts where Latin is required.

    Placeholders, URLs, filenames, filesystem paths, HF cache directory names and
    environment variables are not prose -- they appear verbatim in every language -- so
    counting them would either drown the signal or force them into the allowlist, where
    they would hide real corruption.
    """
    text = re.sub(r"\{[^}]*\}", " ", text)              # {count}, {label}, ...
    text = re.sub(r"\S*\S*/\S*", " ", text)               # /path/to/model, URLs
    text = re.sub(r"\S*--\S*", " ", text)                   # models--org--name
    text = re.sub(r"#[^\s\]]*", " ", text)                  # #{id}
    # Filenames BEFORE the env-var rule: queue_recovery.json would otherwise have its stem
    # eaten first, leaving " .json" with no word character before the dot to match on.
    text = re.sub(
        r"\S+\.(?:json|js|jsx|py|sh|md|txt|ya?ml|html|css|png|jpe?g|safetensors)\b",
        " ", text, flags=re.I,
    )
    text = re.sub(r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b", " ", text)  # env vars
    return set(re.findall(r"[A-Za-z]{2,}", text))


def _node_js(script: str) -> str | None:
    """Run an ESM snippet against the real frontend, with the extensionless loader hook."""
    if not _node_available():
        return None
    frontend = ROOT / "frontend"
    proc = subprocess.run(
        ["node", "--import", str(frontend / "tools" / "register-extensionless.mjs"),
         "--input-type=module", "-e", script],
        cwd=str(frontend / "src" / "i18n"),
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout


def _english_from_node() -> dict | None:
    """key -> English source, read from the real merged catalogue.

    The regex parser in this file misses ~23 keys, and a translation check needs a
    COMPLETE baseline: a key missing from it looks like a translation that invented a
    placeholder.
    """
    out = _node_js(
        'import("./strings.js").then(m => process.stdout.write('
        'JSON.stringify(Object.fromEntries(Object.entries(m.STRINGS)'
        '.map(([k, v]) => [k, v.en])))))'
    )
    return json.loads(out) if out else None


def _overlays() -> dict:
    """Read each lang/<code>.js through node to get the real values."""
    out = {}
    for code in _OVERLAY_CODES:
        raw = _node_js(
            f'import("./lang/{code}.js").then('
            f"m => process.stdout.write(JSON.stringify(m.{code}Strings)))"
        )
        if raw is not None:
            out[code] = json.loads(raw)
    return out


def _node_available() -> bool:
    return shutil.which("node") is not None


def _keys_from_node() -> set | None:
    """The authoritative key list, read out of the real merged object.

    The regex parser above is a reasonable guard on the hand-maintained catalogues, but
    it silently misses ~22 keys, and a scaffold test built on it would compare against
    an incomplete reference. The scaffolds are GENERATED from STRINGS by node anyway, so
    the honest comparison is node's.
    """
    if not _node_available():
        return None
    script = (
        'import("./strings.js").then(m => '
        'process.stdout.write(JSON.stringify(Object.keys(m.STRINGS))))'
    )
    frontend = ROOT / "frontend"
    proc = subprocess.run(
        # The loader hook is REQUIRED, not optional. Without it the app's extensionless
        # imports fail, this returns None, and every caller silently skips.
        [
            "node",
            "--import",
            str(frontend / "tools" / "register-extensionless.mjs"),
            "--input-type=module",
            "-e",
            script,
        ],
        cwd=str(frontend / "src" / "i18n"),
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        return None
    return set(json.loads(proc.stdout))


class TranslationIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.overlays = _overlays()
        cls.english = _english_from_node() or {}
        cls.catalog = _parse_parts()
        cls.catalog.update(_parse_table(STRINGS_FILE))

    def _source(self, key):
        return self.english.get(key, self.catalog.get(key, {}).get("en", ""))

    def test_the_english_baseline_gap_does_not_grow(self):
        """The regex parser in this file does not see every key -- 23 as of this commit.

        That is a pre-existing limitation, not something to assert away, but it means
        every regex-based check here under-verifies. The translation checks therefore
        read the real catalogue through node instead, and this test pins the remaining
        gap so it cannot silently widen. Close it and delete this bound.
        """
        if not _node_available():
            self.skipTest("node not installed")
        self.assertIsNotNone(self.english, "could not read STRINGS via node")
        gap = len(set(self.english) - set(self.catalog))
        self.assertLessEqual(
            gap, 23, f"the regex parser now misses {gap} keys; raise the bound deliberately"
        )

    def test_every_overlay_is_readable(self):
        if not _node_available():
            self.skipTest("node not installed")
        for code in _OVERLAY_CODES:
            with self.subTest(code=code):
                self.assertIn(code, self.overlays, f"lang/{code}.js could not be read")
                self.assertGreater(len(self.overlays[code]), 0)

    def test_placeholders_survive_translation(self):
        """A lost {label} renders literally as "{label}" in the interface.

        This is the highest-value check in this file: the placeholder is the only part of
        a string that MUST survive translation untouched, and it is exactly what breaks
        when a translator rewords a sentence."""
        pattern = re.compile(r"\{(\w+)\}")
        for code, table in self.overlays.items():
            for key, text in table.items():
                source = self._source(key)
                with self.subTest(code=code, key=key):
                    self.assertTrue(source, f"{key} has no English source to compare against")
                    self.assertEqual(
                        sorted(pattern.findall(text)),
                        sorted(pattern.findall(source)),
                        f"{code}:{key} changed the placeholders (en: {source!r})",
                    )

    def test_romance_languages_contain_no_asian_scripts(self):
        for code in ("es", "pt"):
            for key, text in self.overlays.get(code, {}).items():
                with self.subTest(code=code, key=key):
                    for name, rng in _SCRIPT_RANGES.items():
                        self.assertIsNone(
                            re.search(rng, text),
                            f"{code}:{key} contains {name} characters: {text!r}",
                        )

    def test_each_cjk_language_is_written_in_its_own_script(self):
        """Guards against a translation landing in the wrong language, and against the
        corruption mode seen while authoring: a CJK slot filled with another CJK
        language, or with a stray Latin fragment."""
        expect = {"zh": "han", "ja": "kana", "ko": "hangul"}
        forbid = {"zh": ("kana", "hangul"), "ja": ("hangul",), "ko": ("kana",)}
        for code, needed in expect.items():
            for key, text in self.overlays.get(code, {}).items():
                # Pure-technical values legitimately stay Latin.
                latin = _latin_words(text) - _ALLOWED_LATIN
                if not re.search(_SCRIPT_RANGES[needed], text) and not latin:
                    continue
                with self.subTest(code=code, key=key):
                    self.assertTrue(
                        re.search(_SCRIPT_RANGES[needed], text) or not latin,
                        f"{code}:{key} is not {needed}: {text!r}",
                    )
                    for other in forbid[code]:
                        self.assertIsNone(
                            re.search(_SCRIPT_RANGES[other], text),
                            f"{code}:{key} contains {other} script: {text!r}",
                        )

    def test_cjk_values_contain_no_unexplained_latin_words(self):
        """The corruption mode this file exists to catch: while emitting five languages in
        one pass, a fragment of another language lands mid-string. Every Latin word in a
        CJK value must be a known product or vendor name."""
        for code in ("zh", "ja", "ko"):
            for key, text in self.overlays.get(code, {}).items():
                stray = _latin_words(text) - _ALLOWED_LATIN
                with self.subTest(code=code, key=key):
                    self.assertEqual(
                        stray, set(), f"{code}:{key} has unexplained Latin: {sorted(stray)} in {text!r}"
                    )

    def test_no_translation_is_blank_or_placeholder_text(self):
        for code, table in self.overlays.items():
            for key, text in table.items():
                with self.subTest(code=code, key=key):
                    self.assertIsInstance(text, str)
                    self.assertNotEqual(text.strip(), "", f"{code}:{key} is blank")
                    self.assertNotIn(text.strip().lower(), ("tbd", "todo", "n/a", "-"))

    # Strings that are correct in every one of these languages, and so are written out
    # verbatim. Anything added here should be a closed-circuit term, not a sentence.
    _VERBATIM_OK = {
        "licences.copyright",        # "Copyright" is the legal term in all five
        "lora.civitaiTagSuffix",     # "[Civitai #123]" is a literal tag
        "params.ref.inContextBadge", # "FLUX.2 In-Context" is the feature name
        "params.size.baseTitle",     # "Base 512 px"
        "params.size.shapePreviewTitle",  # "{width} x {height}", no words to translate
        "canvas.metaPill",               # pt uses "s" for seconds, as en does
        "generate.civitai.badge",        # "Civitai #123 ↗" is a link label
        "settings.defaults.deepCacheSdxl",   # "DeepCache (SDXL)" is a product name
        "settings.engine.metalCard",         # "Metal / GPU" is a product name
    }

    def test_translations_are_not_verbatim_copies_of_english(self):
        """A copy usually means the slot was filled with English instead of a
        translation. The exceptions are listed in _VERBATIM_OK."""
        identical = []
        for code, table in self.overlays.items():
            for key, text in table.items():
                source = self._source(key)
                if len(source) >= 12 and text == source and key not in self._VERBATIM_OK:
                    identical.append(f"{code}:{key}")
        self.assertEqual(
            identical, [], f"{len(identical)} strings were copied from English: {identical[:10]}"
        )


class ShippedStatusAgreementTests(unittest.TestCase):
    """`status` in languages.js is a claim. This checks the claim against reality.

    It is tempting to let the runtime flip a language to "shipped" once coverage looks
    complete, but then a half-finished translation reclassifies itself and nobody reads
    the diff. The declaration stays a declaration, and this test holds it to account.
    """

    @classmethod
    def setUpClass(cls):
        cls.overlays = _overlays()
        cls.english = _english_from_node() or {}

    def _coverage_gaps(self, code):
        table = self.overlays.get(code, {})
        return {k for k in self.english if not table.get(k)}

    def test_a_shipped_language_has_no_unexplained_gaps(self):
        if not _node_available():
            self.skipTest("node not installed")
        self.assertIsNotNone(self.english, "could not read STRINGS via node")
        for lang in _registry_languages():
            if lang["status"] != "shipped":
                continue
            if lang["code"] not in _OVERLAY_CODES:
                continue  # fr/de/it ship from ./parts/, checked elsewhere in this file
            with self.subTest(code=lang["code"]):
                gaps = self._coverage_gaps(lang["code"]) - _INTENTIONALLY_UNTRANSLATED
                self.assertEqual(
                    gaps, set(), f"{lang['code']} is declared shipped but {len(gaps)} keys are not translated"
                )

    def test_intentionally_untranslated_keys_are_really_untranslated(self):
        """If someone translates one of these, remove it from the allowlist -- otherwise
        the exception quietly stops meaning anything."""
        if not _node_available():
            self.skipTest("node not installed")
        for code, table in self.overlays.items():
            for key in _INTENTIONALLY_UNTRANSLATED:
                with self.subTest(code=code, key=key):
                    if key in table:
                        self.assertNotIn(
                            key, _INTENTIONALLY_UNTRANSLATED,
                            f"{key} now has a translation; drop the exception",
                        )

    def test_coverage_is_reported_so_progress_is_visible(self):
        """A translation project needs a number. Written to lang/coverage.json by the
        build script so CI and the picker can both read it."""
        if not _node_available():
            self.skipTest("node not installed")
        path = ROOT / "frontend" / "src" / "i18n" / "lang" / "coverage.json"
        self.assertTrue(path.exists(), "run `npm run i18n:build`")
        data = json.loads(path.read_text())
        self.assertEqual(sorted(data), sorted(_OVERLAY_CODES))
        for code, entry in data.items():
            with self.subTest(code=code):
                self.assertEqual(entry["done"], len(self.english) - len(
                    _INTENTIONALLY_UNTRANSLATED & set(self.english)
                ))
