"""Guards the translation table.

A missing or empty language used to be invisible: the UI renders whatever is in the
table, so a key that exists for French but not German shows an empty label in German.
This fails the build instead.

The table is parsed with a regex rather than a JS parser on purpose -- there is no
node-based test runner in this repo, and the file's shape (a flat object of string
literals) is stable enough to check without executing JavaScript.
"""

import re
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
