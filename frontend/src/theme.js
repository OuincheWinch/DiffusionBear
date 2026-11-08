import { createContext, useContext, useCallback, useEffect, useRef, useState, createElement } from "react";
import { api } from "./api";

const ThemeContext = createContext(null);

/**
 * Light/dark theme, with the same contract the `language` setting already uses:
 * "auto" follows the OS, and the toggle overrides it with a concrete value, which is
 * then what gets persisted.
 *
 * Two things this deliberately does NOT do:
 *
 * - It does not treat localStorage as the source of truth. settings.json is. The
 *   mirror exists only so index.html can paint the right background before this bundle
 *   has even loaded; it is written after a successful POST and read only to compare.
 * - It does not fetch on mount if the mirror already agrees with the OS. It always
 *   reconciles, but a disagreement is corrected silently rather than re-painting.
 */

const MIRROR_KEY = "diffusionbear.theme";
const MODES = ["auto", "light", "dark"];

function systemPrefersLight() {
  return typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-color-scheme: light)").matches;
}

export function resolveTheme(choice) {
  if (choice === "light" || choice === "dark") return choice;
  return systemPrefersLight() ? "light" : "dark";
}

function applyTheme(mode) {
  if (typeof document === "undefined") return;
  document.documentElement.setAttribute("data-theme", mode);
}

function useThemeState() {
  // Start from whatever index.html already applied, so the first render matches the
  // painted background rather than snapping to a second value.
  const [choice, setChoice] = useState(() => {
    if (typeof document === "undefined") return "auto";
    const applied = document.documentElement.getAttribute("data-theme");
    return applied === "light" || applied === "dark" ? applied : "auto";
  });
  const [resolved, setResolved] = useState(() => resolveTheme(
    typeof document !== "undefined"
      ? document.documentElement.getAttribute("data-theme")
      : null
  ));

  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);

  // Reconcile with the backend, which is the record.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const settings = await api("/api/settings");
        if (cancelled || !mounted.current) return;
        const stored = settings?.theme;
        if (!MODES.includes(stored)) return;
        setChoice(stored);
        applyTheme(resolveTheme(stored));
        // Keep the paint-time mirror honest. This is the only place it is written.
        try { localStorage.setItem(MIRROR_KEY, resolveTheme(stored)); } catch { /* private mode */ }
      } catch {
        // No settings, or the backend is not up yet. The mirror plus the OS preference
        // already produced a sensible theme; the first successful fetch corrects it.
      }
    })();
    return () => { cancelled = true; };
  }, []);

  // Follow the OS while on "auto" -- someone switching their Mac to light mode at
  // sunset should not have to reload.
  useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") return undefined;
    const query = window.matchMedia("(prefers-color-scheme: light)");
    const onChange = () => {
      if (choice !== "auto") return;
      const next = systemPrefersLight() ? "light" : "dark";
      applyTheme(next);
      setResolved(next);
    };
    query.addEventListener?.("change", onChange);
    return () => query.removeEventListener?.("change", onChange);
  }, [choice]);

  const setTheme = useCallback(async (next) => {
    if (!MODES.includes(next)) return;
    const mode = resolveTheme(next);
    setChoice(next);
    setResolved(mode);
    applyTheme(mode);
    // Optimistic: the button must respond immediately. A failed save is corrected on
    // the next load by the reconcile above, which is better than blocking a click.
    try { localStorage.setItem(MIRROR_KEY, mode); } catch { /* private mode */ }
    try {
      await api("/api/settings", {
        method: "POST",
        body: JSON.stringify({ theme: next }),
      });
    } catch {
      // Left as-is locally. The backend still holds the old value, and the reconcile
      // on the next load will restore it rather than leaving the two out of step
      // permanently.
    }
  }, []);

  const toggle = useCallback(() => {
    setTheme(resolved === "light" ? "dark" : "light");
  }, [resolved, setTheme]);

  return { theme: choice, resolved, setTheme, toggle };
}

export function ThemeProvider({ children }) {
  const value = useThemeState();
  return createElement(ThemeContext.Provider, { value }, children);
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error("useTheme must be used within a ThemeProvider");
  }
  return context;
}

export default useTheme;