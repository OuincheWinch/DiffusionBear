import { useState } from "react";
import TokenManager from "./TokenManager";
import DefaultsSection from "./settings/DefaultsSection";
import EngineSection from "./settings/EngineSection";
import WorkspaceSection from "./settings/WorkspaceSection";
import SamplersSection from "./settings/SamplersSection";
import QueueSection from "./settings/QueueSection";
import EnhancerSystemSection from "./settings/EnhancerSystemSection";
import LanguageSection from "./settings/LanguageSection";
import { useI18n } from "../i18n/I18nContext";
import LicencesTab from "./LicencesTab";
import { useSettings } from "../hooks/useSettings";
import { useModelsList } from "../hooks/useModelsList";

export default function ParametersTab({ onNavigate, subtab: controlledSubtab, onSubtabChange }) {
  const { t } = useI18n();
  const { settings, loading: settingsLoading, update, refresh } = useSettings();
  const { models } = useModelsList();
  const [feedback, setFeedback] = useState(null);
  const [tokenAutofocus, setTokenAutofocus] = useState(null);
  const [internalSubtab, setInternalSubtab] = useState("prefs");
  const subtab = controlledSubtab !== undefined ? controlledSubtab : internalSubtab;
  const setSubtab = onSubtabChange || setInternalSubtab;

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
          {feedback.type === "error" ? (
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{ verticalAlign: "-2px", marginRight: "6px" }}><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
          ) : (
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{ verticalAlign: "-2px", marginRight: "6px" }}><polyline points="20 6 9 17 4 12"/></svg>
          )}
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
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>
          {t("settings.tab.preferences")}
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={subtab === "enhancer"}
          className={`params-subtab${subtab === "enhancer" ? " active" : ""}`}
          onClick={() => setSubtab("enhancer")}
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z"/></svg>
          {t("settings.tab.enhancer")}
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={subtab === "licences"}
          className={`params-subtab${subtab === "licences" ? " active" : ""}`}
          onClick={() => setSubtab("licences")}
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
          {t("licences.title")}
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
        <h3>{t("settings.section.workspaceTitle")}</h3>
        <p className="params-section-desc">{t("settings.section.workspaceDesc")}</p>
        <WorkspaceSection onFeedback={handleFeedback} />
      </section>

      <section className="params-section">
        <h3>{t("settings.section.samplersTitle")}</h3>
        <p className="params-section-desc">{t("settings.section.samplersDesc")}</p>
        <SamplersSection />
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