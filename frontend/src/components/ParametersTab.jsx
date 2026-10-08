import { useState } from "react";
import TokenManager from "./TokenManager";
import DefaultsSection from "./settings/DefaultsSection";
import EngineSection from "./settings/EngineSection";
import QueueSection from "./settings/QueueSection";
import EnhancerSystemSection from "./settings/EnhancerSystemSection";
import LanguageSection from "./settings/LanguageSection";
import { useI18n } from "../i18n/I18nContext";
import LicencesTab from "./LicencesTab";
import { useSettings } from "../hooks/useSettings";
import { useModelsList } from "../hooks/useModelsList";

export default function ParametersTab({ onNavigate }) {
  const { t } = useI18n();
  const { settings, loading: settingsLoading, update, refresh } = useSettings();
  const { models } = useModelsList();
  const [feedback, setFeedback] = useState(null);
  const [tokenAutofocus, setTokenAutofocus] = useState(null);
  const [subtab, setSubtab] = useState("prefs");

  function handleFeedback(fb) {
    setFeedback(fb);
    refresh();
    if (fb?.type === "success") {
      setTimeout(() => setFeedback(null), 4000);
    }
  }

  return (
    <div className="parameters-tab">
      {feedback && (
        <div className={`settings-feedback ${feedback.type}`} role="status">
          {feedback.type === "error" ? "⚠ " : "✓ "}
          {feedback.text}
        </div>
      )}

      {/* Language first: it is the one setting a user may need before they can
          read anything else, and it used to sit below the fold at the very bottom
          of a long scroll. */}
      <LanguageSection />

      <div className="params-subtabs" role="tablist">
        <button
          type="button"
          role="tab"
          aria-selected={subtab === "prefs"}
          className={`params-subtab${subtab === "prefs" ? " active" : ""}`}
          onClick={() => setSubtab("prefs")}
        >
          ⚙️ {t("settings.tab.preferences")}
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={subtab === "enhancer"}
          className={`params-subtab${subtab === "enhancer" ? " active" : ""}`}
          onClick={() => setSubtab("enhancer")}
        >
          🧠 {t("settings.tab.enhancer")}
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={subtab === "licences"}
          className={`params-subtab${subtab === "licences" ? " active" : ""}`}
          onClick={() => setSubtab("licences")}
        >
          ⚖ {t("licences.title")}
        </button>
      </div>

      {subtab === "licences" ? (
        <LicencesTab />
      ) : subtab === "enhancer" ? (
        <section className="params-section">
          <h3>{t("settings.tab.enhancerTitle")}</h3>
          <p className="params-section-desc">{t("settings.tab.enhancerDesc")}</p>
          <EnhancerSystemSection onFeedback={handleFeedback} onSaved={refresh} />
        </section>
      ) : (
        <>
      <section className="params-section">
        <h3>{t("settings.section.defaultsTitle")}</h3>
        <p className="params-section-desc">{t("settings.section.defaultsDesc")}</p>
        {settingsLoading ? (
          <p className="hint">{t("app.loading")}</p>
        ) : (
          <DefaultsSection
            settings={settings}
            models={models}
            update={update}
            onFeedback={handleFeedback}
          />
        )}
      </section>

      <section className="params-section">
        <h3>{t("settings.section.engineTitle")}</h3>
        <p className="params-section-desc">{t("settings.section.engineDesc")}</p>
        <EngineSection onFeedback={handleFeedback} />
      </section>

      <section className="params-section">
        <h3>{t("settings.section.queueTitle")}</h3>
        <p className="params-section-desc">{t("settings.section.queueDesc")}</p>
        <QueueSection onNavigate={onNavigate} />
      </section>

      <section className="params-section">
        <h3>{t("settings.section.secretsTitle")}</h3>
        <p className="params-section-desc">
          {t("settings.section.secretsDescBefore")} <code>backend/data/</code>{" "}
          ({t("settings.section.secretsDescAfter")} <code>civitai_token.txt</code>,{" "}
          <code>hf_token.txt</code>). {t("settings.section.secretsDescTail")}
        </p>
        <TokenManager
          autofocus={tokenAutofocus}
          onTokenSaved={() => setTokenAutofocus(null)}
        />
        <p className="params-hint">
          {t("settings.queue.tokenFallback")} <code>~/.cache/huggingface/token</code>
        </p>
      </section>
        </>
      )}
    </div>
  );
}