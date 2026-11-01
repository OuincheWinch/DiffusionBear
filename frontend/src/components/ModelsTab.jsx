import { useEffect, useRef, useState } from "react";
import SourceBrowser from "./SourceBrowser";
import ImportTab from "./ImportTab";
import OnboardModelTab from "./OnboardModelTab";
import ModelsSection from "./settings/ModelsSection";
import StorageSection from "./settings/StorageSection";
import HfCacheSection from "./settings/HfCacheSection";
import InventoryTable from "./InventoryTable";
import { useI18n } from "../i18n/I18nContext";
import { useModelsList } from "../hooks/useModelsList";

/**
 * The Models primary tab.
 *
 * Segmented into five focused views:
 * - "installed": Active configured models in DiffusionBear
 * - "browse": Hugging Face discovery and model downloader
 * - "detected": Local unattached/orphan models found on disk (Inventory)
 * - "import": External imports (Draw Things, DiffusionBee) and manual path registration
 * - "storage": Storage disk space and Hugging Face cache cleanup
 *
 * All sub-tabs stay mounted with display:none to preserve filters and in-flight operations.
 */
export default function ModelsTab({ onNavigate }) {
  const { t } = useI18n();
  const [side, setSide] = useState("installed");
  const { models, loading: modelsLoading, refresh: refreshModels } = useModelsList();
  const [feedback, setFeedback] = useState(null);
  const feedbackTimer = useRef(null);

  useEffect(() => () => {
    if (feedbackTimer.current != null) clearTimeout(feedbackTimer.current);
  }, []);

  function handleFeedback(fb) {
    setFeedback(fb);
    if (feedbackTimer.current != null) clearTimeout(feedbackTimer.current);
    if (fb?.type === "success") {
      refreshModels();
      feedbackTimer.current = setTimeout(() => setFeedback(null), 4000);
    }
    return undefined;
  }

  const tabItems = [
    {
      key: "installed",
      label: t("models.tab.installed"),
      icon: (
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
          <polyline points="3.27 6.96 12 12.01 20.73 6.96"/>
          <line x1="12" y1="22.08" x2="12" y2="12"/>
        </svg>
      ),
    },
    {
      key: "browse",
      label: t("models.tab.browse"),
      icon: (
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <circle cx="12" cy="12" r="10"/>
          <polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/>
        </svg>
      ),
    },
    {
      key: "detected",
      label: t("models.tab.detected"),
      icon: (
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <line x1="22" y1="12" x2="2" y2="12"/>
          <path d="M5.45 5.11L2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/>
          <line x1="6" y1="16" x2="6.01" y2="16"/>
          <line x1="10" y1="16" x2="10.01" y2="16"/>
        </svg>
      ),
    },
    {
      key: "import",
      label: t("models.tab.import"),
      icon: (
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="7 10 12 15 17 10"/>
          <line x1="12" y1="15" x2="12" y2="3"/>
        </svg>
      ),
    },
    {
      key: "storage",
      label: t("models.tab.storage"),
      icon: (
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <ellipse cx="12" cy="5" rx="9" ry="3"/>
          <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/>
          <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>
        </svg>
      ),
    },
  ];

  return (
    <div className="parameters-tab models-tab">
      {feedback && (
        <div className={`settings-feedback ${feedback.type}`} role="status">
          {feedback.type === "error" ? (
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{ verticalAlign: "-2px", marginRight: "6px" }}><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
          ) : (
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{ verticalAlign: "-2px", marginRight: "6px" }}><polyline points="20 6 9 17 4 12"/></svg>
          )}
          {feedback.text}
        </div>
      )}

      <div className="models-tab-split" role="tablist">
        {tabItems.map((tab) => (
          <button
            key={tab.key}
            type="button"
            role="tab"
            aria-selected={side === tab.key}
            className={`models-tab-split-btn${side === tab.key ? " active" : ""}`}
            onClick={() => setSide(tab.key)}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      <div style={{ display: side === "installed" ? "block" : "none" }}>
        <section className="params-section">
          <h3>{t("settings.section.modelsTitle")}</h3>
          <p className="params-section-desc">{t("settings.section.modelsDesc")}</p>
          <ModelsSection
            models={models}
            loading={modelsLoading}
            onModelsChanged={refreshModels}
            onFeedback={handleFeedback}
          />
        </section>
      </div>

      <div style={{ display: side === "browse" ? "block" : "none" }}>
        <SourceBrowser onInstalled={refreshModels} />
      </div>

      <div style={{ display: side === "detected" ? "block" : "none" }}>
        <InventoryTable models={models} onModelsChanged={refreshModels} onFeedback={handleFeedback} />
      </div>

      <div style={{ display: side === "import" ? "block" : "none" }}>
        <ImportTab onImported={refreshModels} />
        <OnboardModelTab onRegistered={refreshModels} />
      </div>

      <div style={{ display: side === "storage" ? "block" : "none" }}>
        <section className="params-section">
          <h3>{t("settings.section.storageTitle")}</h3>
          <p className="params-section-desc">{t("settings.section.storageDesc")}</p>
          <StorageSection onFeedback={handleFeedback} />
        </section>

        <section className="params-section">
          <h3>{t("settings.section.hfCacheTitle")}</h3>
          <p className="params-section-desc">{t("settings.section.hfCacheDesc")}</p>
          <HfCacheSection onFeedback={handleFeedback} />
        </section>
      </div>

      {onNavigate && (
        <button
          type="button"
          className="btn-mini models-tab-goto-generate"
          onClick={() => onNavigate("generate")}
        >
          {t("app.generate")}
        </button>
      )}
    </div>
  );
}
