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
 * Model management, Storage and the Hugging Face cache used to be vertical sections inside
 * Parameters > Preferences, sandwiched between Defaults and Engine. That was the wrong
 * home for them: they are inventory, not preferences, they are the things a user opens the
 * app to *look at*, and burying them below generation defaults meant a model download took
 * a scroll and a hunt to find. Parameters keeps what actually changes behaviour.
 *
 * The HF browser sits at the top on purpose. Installing a model is the entry point to
 * everything else here -- until something is installed, the registry, storage and cache
 * tables below have little to say.
 */
export default function ModelsTab({ onNavigate }) {
  const { t } = useI18n();
  // Installed vs Import. A useState string, not a router: this is a sub-section of one
  // tab, and the inactive side stays mounted with display:none like every other view in
  // this app. Unmounting it would throw away the browse filters and the in-flight
  // conversion the moment the user glances at the other side.
  const [side, setSide] = useState("installed");
  const { models, loading: modelsLoading, refresh: refreshModels } = useModelsList();
  const [feedback, setFeedback] = useState(null);
  // A cleanup function returned from an event handler is inert -- React only
  // runs cleanups returned from useEffect -- so the old `return () =>
  // clearTimeout(timer)` never fired. The timer lives in a ref and is
  // cleared on unmount and on every new feedback instead.
  const feedbackTimer = useRef(null);

  useEffect(() => () => {
    if (feedbackTimer.current != null) clearTimeout(feedbackTimer.current);
  }, []);

  function handleFeedback(fb) {
    setFeedback(fb);
    if (feedbackTimer.current != null) clearTimeout(feedbackTimer.current);
    if (fb?.type === "success") {
      // Keep the registry table honest for a moment after a change, then get out of the way.
      refreshModels();
      feedbackTimer.current = setTimeout(() => setFeedback(null), 4000);
    }
    return undefined;
  }

  return (
    <div className="parameters-tab models-tab">
      {feedback && (
        <div className={`settings-feedback ${feedback.type}`} role="status">
          {feedback.type === "error" ? "⚠ " : "✓ "}
          {feedback.text}
        </div>
      )}

      <div className="models-tab-split" role="tablist">
        {[["installed", "sources.installed"], ["import", "sources.importTab"]].map(([key, labelKey]) => (
          <button
            key={key}
            type="button"
            role="tab"
            aria-selected={side === key}
            className={`models-tab-split-btn${side === key ? " active" : ""}`}
            onClick={() => setSide(key)}
          >
            {key === "installed" ? (
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>
            ) : (
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
            )}
            {t(labelKey)}
          </button>
        ))}
      </div>

      <div style={{ display: side === "installed" ? "block" : "none" }}>
      <SourceBrowser onInstalled={refreshModels} />

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

      <InventoryTable onModelsChanged={refreshModels} onFeedback={handleFeedback} />
      </div>

      <div style={{ display: side === "import" ? "block" : "none" }}>
        <ImportTab onImported={refreshModels} />
        <OnboardModelTab onRegistered={refreshModels} />
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