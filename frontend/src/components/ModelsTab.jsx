import { useState } from "react";
import HfModelBrowser from "./HfModelBrowser";
import ModelsSection from "./settings/ModelsSection";
import StorageSection from "./settings/StorageSection";
import HfCacheSection from "./settings/HfCacheSection";
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
  const { models, loading: modelsLoading, refresh: refreshModels } = useModelsList();
  const [feedback, setFeedback] = useState(null);

  function handleFeedback(fb) {
    setFeedback(fb);
    if (fb?.type === "success") {
      // Keep the registry table honest for a moment after a change, then get out of the way.
      refreshModels();
      const timer = setTimeout(() => setFeedback(null), 4000);
      return () => clearTimeout(timer);
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

      <HfModelBrowser onInstalled={refreshModels} />

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