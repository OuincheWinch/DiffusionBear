// The four shipped interface languages. Codes are plain BCP-47-ish tags so they
// double as the <html lang> value and as the persisted settings value.
export const LANGUAGES = [
  { code: "en", label: "English", flag: "🇬🇧" },
  { code: "fr", label: "Français", flag: "🇫🇷" },
  { code: "de", label: "Deutsch", flag: "🇩🇪" },
  { code: "it", label: "Italiano", flag: "🇮🇹" },
];

export const DEFAULT_LANGUAGE = "en";

export const LANGUAGE_CODES = LANGUAGES.map((l) => l.code);

/** Resolve a BCP-47 tag ("fr-CA", "de_AT", "it") to a supported code. */
export function normaliseLanguage(tag) {
  if (!tag) return null;
  const base = String(tag).toLowerCase().split(/[-_]/)[0];
  return LANGUAGE_CODES.includes(base) ? base : null;
}

/**
 * Pick a language for first launch: the stored preference, else the OS locale.
 * Anything unsupported falls back to English rather than rendering raw keys.
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
    if (fromNav) return fromNav;
  }
  return DEFAULT_LANGUAGE;
}
