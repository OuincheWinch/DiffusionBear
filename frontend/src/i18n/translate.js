import { STRINGS } from "./strings";
import { DEFAULT_LANGUAGE, detectLanguage } from "./languages";

/**
 * Translator for code outside the React tree (native drop handler, error pages).
 * It always resolves against the OS locale, since there is no provider to ask.
 * Inside components use useI18n() so the value follows a live language change.
 */
export function t(key, vars) {
  const lang = detectLanguage(null);
  const entry = STRINGS[key];
  let text = entry ? (entry[lang] ?? entry[DEFAULT_LANGUAGE] ?? key) : key;
  if (!vars) return text;
  return text.replace(/\{(\w+)\}/g, (match, name) =>
    Object.prototype.hasOwnProperty.call(vars, name) ? String(vars[name]) : match
  );
}
