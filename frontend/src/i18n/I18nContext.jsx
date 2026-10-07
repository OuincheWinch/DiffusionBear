import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { STRINGS } from "./strings";
import { DEFAULT_LANGUAGE, detectLanguage } from "./languages";
import { api } from "../api";

/**
 * Look up a string, interpolating {name} placeholders.
 *
 * A missing key returns the key itself rather than an empty string, so a gap is
 * visible during review instead of rendering as a blank label. `test_i18n.py`
 * exists to stop that from shipping: it fails if any key is missing a language.
 */
function lookup(key, lang, vars) {
  const entry = STRINGS[key];
  let text;
  if (!entry) {
    text = key;
  } else {
    text = entry[lang] ?? entry[DEFAULT_LANGUAGE] ?? key;
  }
  if (!vars) return text;
  return text.replace(/\{(\w+)\}/g, (match, name) =>
    Object.prototype.hasOwnProperty.call(vars, name) ? String(vars[name]) : match
  );
}

const I18nContext = createContext(null);

export function I18nProvider({ children }) {
  const [lang, setLangState] = useState(() => detectLanguage(null));
  // The raw preference ("auto" or a concrete code). Kept apart from `lang`, which
  // is the resolved code actually used to translate: the picker has to show which
  // option is selected, and "auto" has no resolved code of its own.
  const [preference, setPreference] = useState("auto");

  // The stored preference is the source of truth on first paint, but it arrives
  // asynchronously. Read it once on mount and adopt it unless the user has
  // already picked a language in this session.
  useEffect(() => {
    let cancelled = false;
    api("/api/settings")
      .then((settings) => {
        if (cancelled) return;
        const stored = settings?.language;
        if (stored) {
          setPreference(stored);
          setLangState(detectLanguage(stored));
        }
      })
      .catch(() => {
        // A backend that is not up yet is normal on first launch; the OS locale
        // already picked above, so there is nothing to recover from.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (typeof document !== "undefined") {
      document.documentElement.lang = lang;
    }
  }, [lang]);

  const setLang = useCallback(async (next) => {
    setPreference(next);
    setLangState(detectLanguage(next));
    try {
      await api("/api/settings", {
        method: "POST",
        body: JSON.stringify({ language: next }),
      });
    } catch (err) {
      // The language still applies for this session; only persistence failed.
      console.error("[DiffusionBear] could not save language:", err);
    }
  }, []);

  const t = useCallback(
    (key, vars) => lookup(key, lang, vars),
    [lang]
  );

  const value = useMemo(
    () => ({ lang, preference, setLang, t }),
    [lang, preference, setLang, t]
  );
  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n() {
  const ctx = useContext(I18nContext);
  if (!ctx) {
    throw new Error("useI18n must be used inside <I18nProvider>");
  }
  return ctx;
}
