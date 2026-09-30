import { useI18n } from "../../i18n/I18nContext";
import { LANGUAGES } from "../../i18n/languages";

/**
 * Interface language picker.
 *
 * "auto" follows the OS locale. Choosing a concrete code persists it through
 * /api/settings, so the choice survives a reinstall; the backend validates the
 * value against the same four codes and rejects anything else.
 *
 * The language names are intentionally NOT translated: a language picker that
 * renames its own entries based on the current selection makes the list harder to
 * find. Each language is shown in its own language, with its flag as the anchor.
 */
export default function LanguageSection() {
  const { preference, setLang, t } = useI18n();

  return (
    <section className="params-section language-section">
      <h3>🌐 {t("language.title")}</h3>
      <div className="language-grid" role="group" aria-label={t("language.title")}>
        <button
          type="button"
          className={`language-option ${preference === "auto" ? "active" : ""}`}
          onClick={() => setLang("auto")}
          aria-pressed={preference === "auto"}
          title={t("language.auto")}
        >
          <span className="language-flag" aria-hidden="true">🌐</span>
          <span className="language-name">{t("language.auto")}</span>
        </button>
        {LANGUAGES.map((entry) => (
          <button
            key={entry.code}
            type="button"
            className={`language-option ${preference === entry.code ? "active" : ""}`}
            onClick={() => setLang(entry.code)}
            aria-pressed={preference === entry.code}
          >
            <span className="language-flag" aria-hidden="true">{entry.flag}</span>
            <span className="language-name">{entry.label}</span>
          </button>
        ))}
      </div>
    </section>
  );
}
