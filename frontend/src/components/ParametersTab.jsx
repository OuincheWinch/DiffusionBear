import { useCallback, useEffect, useState } from "react";
import { api } from "../api";
import TokenManager from "./TokenManager";
import DefaultsSection from "./settings/DefaultsSection";
import EngineSection from "./settings/EngineSection";
import ModelsSection from "./settings/ModelsSection";
import HfCacheSection from "./settings/HfCacheSection";
import QueueSection from "./settings/QueueSection";
import StorageSection from "./settings/StorageSection";
import EnhancerSystemSection from "./settings/EnhancerSystemSection";
import { useSettings } from "../hooks/useSettings";

export default function ParametersTab({ onNavigate }) {
  const { settings, loading: settingsLoading, update, refresh } = useSettings();
  const [models, setModels] = useState([]);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [feedback, setFeedback] = useState(null);
  const [tokenAutofocus, setTokenAutofocus] = useState(null);
  const [subtab, setSubtab] = useState("prefs");

  const refreshModels = useCallback(async () => {
    setModelsLoading(true);
    try {
      const list = await api("/api/models");
      setModels(list);
    } catch {
      /* transient */
    } finally {
      setModelsLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshModels();
  }, [refreshModels]);

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

      <div className="params-subtabs" role="tablist">
        <button
          type="button"
          role="tab"
          aria-selected={subtab === "prefs"}
          className={`params-subtab${subtab === "prefs" ? " active" : ""}`}
          onClick={() => setSubtab("prefs")}
        >
          ⚙️ Preferences
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={subtab === "enhancer"}
          className={`params-subtab${subtab === "enhancer" ? " active" : ""}`}
          onClick={() => setSubtab("enhancer")}
        >
          🧠 Prompt Enhancer Prompts (experimental)
        </button>
      </div>

      {subtab === "enhancer" ? (
        <section className="params-section">
          <h3>🧠 Prompt Enhancer — System Prompts (experimental)</h3>
          <p className="params-section-desc">
            Customize the system prompt the local LLM (Qwen2.5-0.5B-Instruct via MLX) uses when
            you click ✨ Enhance. One editable prompt per engine — FLUX.2 Klein, SDXL Lightning,
            Krea 2 Turbo and Z-Image Turbo. Saved overrides are used immediately by the enhancer.
            The enforced contract is prompt-only output within each engine's length cap (no
            preamble, no explanation), but the feature itself is experimental.
          </p>
          <EnhancerSystemSection onFeedback={handleFeedback} onSaved={refresh} />
        </section>
      ) : (
        <>
      <section className="params-section">
        <h3>🖋 Defaults &amp; Personalization</h3>
        <p className="params-section-desc">
          Default generation preferences for new images, plus the artist credit embedded in
          every output. Your artist name replaces the previous hard-coded credit — great for a
          public release.
        </p>
        {settingsLoading ? (
          <p className="hint">Loading preferences…</p>
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
        <h3>🖥 Engine &amp; GPU</h3>
        <p className="params-section-desc">
          Live Metal usage, wired-memory budgets, resident mflux/SDXL pipelines and the
          idle auto-release countdown.
        </p>
        <EngineSection onFeedback={handleFeedback} />
      </section>

      <section className="params-section">
        <h3>🗂 Model Management</h3>
        <p className="params-section-desc">
          Installed status and disk footprint of every engine. Remove weights to free space;
          they are re-downloaded on demand.
        </p>
        <ModelsSection
          models={models}
          loading={modelsLoading}
          onModelsChanged={refreshModels}
          onFeedback={handleFeedback}
        />
      </section>

      <section className="params-section">
        <h3>⏳ Queue &amp; Pending Jobs</h3>
        <p className="params-section-desc">
          Watch the generation queue and re-queue prompts that were interrupted or cancelled.
        </p>
        <QueueSection onNavigate={onNavigate} />
      </section>

        <section className="params-section">
          <h3>📦 Storage</h3>
          <p className="params-section-desc">
            What is on disk and what accounts for it. Read-only: this panel never deletes anything,
            and never marks anything reclaimable.
          </p>
          <StorageSection onFeedback={handleFeedback} />
        </section>

      <section className="params-section">
        <h3>💾 Hugging Face Cache</h3>
        <p className="params-section-desc">
          Local copy of every downloaded model repo. Clear entries to reclaim disk space.
        </p>
        <HfCacheSection onFeedback={handleFeedback} />
      </section>

      <section className="params-section">
        <h3>🔐 Secret Management</h3>
        <p className="params-section-desc">
          API keys &amp; tokens are saved by the backend into local files under{" "}
          <code>backend/data/</code> (e.g. <code>civitai_token.txt</code>,{" "}
          <code>hf_token.txt</code>). They are never sent to the browser clients, never logged,
          and never exposed by the API — they are used only server-side to authenticate outbound
          requests to Civitai / Hugging Face.
        </p>
        <TokenManager
          autofocus={tokenAutofocus}
          onTokenSaved={() => setTokenAutofocus(null)}
        />
        <p className="params-hint">
          Note: gated Hugging Face repos also fall back to a token stored in{" "}
          <code>~/.cache/huggingface/token</code> if this file is empty.
        </p>
      </section>
        </>
      )}
    </div>
  );
}