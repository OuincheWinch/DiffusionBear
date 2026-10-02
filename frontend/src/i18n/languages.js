// Every interface language, in picker order. Codes are plain BCP-47-ish tags so they
// double as the <html lang> value and as the persisted settings value.
//
// status is "shipped" once a catalogue actually exists for the language, and
// "scaffold" while every string is still null. The two are NOT interchangeable:
//
//   * A scaffold language renders through the fallback chain in translate.js, so
//     picking one today shows the English text. That is the honest state of the
//     translation and it is why these are still selectable.
//   * detectLanguage() refuses to AUTO-select a scaffold language, because a
//     Spanish-locale Mac that silently rendered English under a picker claiming
//     "Español" would look broken rather than untranslated. An explicit choice
//     is allowed, since that is the user testing or forcing the language.
export const LANGUAGES = [
  { code: "en", label: "English", flag: "🇬🇧", status: "shipped" },
  { code: "fr", label: "Français", flag: "🇫🇷", status: "shipped" },
  { code: "de", label: "Deutsch", flag: "🇩🇪", status: "shipped" },
  { code: "it", label: "Italiano", flag: "🇮🇹", status: "shipped" },
  { code: "es", label: "Español", flag: "🇪🇸", status: "scaffold" },
  { code: "zh", label: "简体中文", flag: "🇨🇳", status: "scaffold" },
  { code: "ja", label: "日本語", flag: "🇯🇵", status: "scaffold" },
  { code: "pt", label: "Português", flag: "🇵🇹", status: "scaffold" },
  { code: "ko", label: "한국어", flag: "🇰🇷", status: "scaffold" },
];

/** Languages with a real catalogue; everything else falls back to English. */
export const TRANSLATED_LANGUAGES = LANGUAGES.filter((l) => l.status === "shipped");

/** Language codes awaiting translation, one scaffold file each in ./scaffold/. */
export const SCAFFOLD_LANGUAGES = LANGUAGES.filter((l) => l.status === "scaffold");

export const DEFAULT_LANGUAGE = "en";

export const LANGUAGE_CODES = LANGUAGES.map((l) => l.code);

/** Codes that must match, in order, the validation in backend/routers/settings.py
 *  and backend/app_settings.py. A language added here and not there is silently
 *  rejected on save. */
export const ALL_LANGUAGE_CODES = ["auto", ...LANGUAGE_CODES];

/** Resolve a BCP-47 tag ("fr-CA", "de_AT", "it") to a supported code. */
export function normaliseLanguage(tag) {
  if (!tag) return null;
  const base = String(tag).toLowerCase().split(/[-_]/)[0];
  return LANGUAGE_CODES.includes(base) ? base : null;
}

/**
 * Pick a language for first launch: the stored preference, else the OS locale.
 * Anything unsupported falls back to English rather than rendering raw keys. A
 * locale with no catalogue yet (see status in LANGUAGES) also falls back, so the OS
 * never lands the user in an interface that is really just English.
 */
export function detectLanguage(stored) {
  if (stored && stored !== "auto") {
    const explicit = normaliseLanguage(stored);
    if (explicit) return explicit;
  }
  if (stored === "auto") {
    const fromNav = normaliseLanguage(
      typeof navigator !== "undefined" ? navigator.language : ""
    );
    if (fromNav && TRANSLATED_LANGUAGES.some((l) => l.code === fromNav)) return fromNav;
  }
  return DEFAULT_LANGUAGE;
}
