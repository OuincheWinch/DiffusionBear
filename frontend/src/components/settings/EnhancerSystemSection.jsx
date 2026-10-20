import { useEffect, useMemo, useState } from "react";
import { api } from "../../api";
import { useI18n } from "../../i18n/I18nContext";

const API = "/api/prompt/enhancer/system-prompts";

// The editable box is pre-filled with the user's override if present, otherwise
// with the built-in instructions — so the model's system prompt is always
// visible and editable rather than hidden behind a "view built-in" toggle.
// Text mode shows the engine guidance; JSON mode shows the full JSON contract
// (engine guidance + schema structure + fill-in guidelines).
function draftFor(engine, mode) {
  if (mode === "json") {
    return (
      engine.custom_json_instructions || engine.default_json_instructions || ""
    );
  }
  return engine.custom_instructions || engine.default_instructions || "";
}

function isCustom(engine, mode) {
  return mode === "json" ? !!engine.is_json_custom : !!engine.is_custom;
}

export default function EnhancerSystemSection({ onFeedback, onSaved }) {
  const { t } = useI18n();
  const [engines, setEngines] = useState([]);
  const [models, setModels] = useState([]);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [downloadingId, setDownloadingId] = useState(null);
  const [switchingId, setSwitchingId] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeKey, setActiveKey] = useState(null);
  const [mode, setMode] = useState("text");
  const [drafts, setDrafts] = useState({});
  const [saving, setSaving] = useState(false);
  const [showPreview, setShowPreview] = useState(false);
  const [showDefault, setShowDefault] = useState(false);

  async function load() {
    setLoading(true);
    try {
      const data = await api(API);
      const list = data.engines || [];
      setEngines(list);
      const next = {};
      for (const e of list) {
        next[`text:${e.key}`] = draftFor(e, "text");
        next[`json:${e.key}`] = draftFor(e, "json");
      }
      setDrafts(next);
      setActiveKey((prev) => prev || list[0]?.key || null);
      setError(null);
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
    loadModels();
    // loadModels is a per-render closure over onFeedback; depending on it
    // would refire the fetch every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function loadModels() {
    setModelsLoading(true);
    try {
      const data = await api("/api/prompt/enhancer/models");
      setModels(Array.isArray(data?.models) ? data.models : []);
    } catch (e) {
      onFeedback?.({ type: "error", text: e.message || String(e) });
    } finally {
      setModelsLoading(false);
    }
  }

  async function downloadModel(id) {
    setDownloadingId(id);
    try {
      await api("/api/prompt/enhancer/download", {
        method: "POST",
        body: JSON.stringify({ id }),
      });
      await loadModels();
      onFeedback?.({ type: "success", text: t("settings.enhancer.downloadedBadge") });
    } catch (e) {
      onFeedback?.({ type: "error", text: e.message || String(e) });
    } finally {
      setDownloadingId(null);
    }
  }

  async function activateModel(id, cached) {
    // Switching to an uncached model downloads it first: activating weights
    // that are not on disk would only fail later, inside Enhance.
    if (!cached) {
      await downloadModel(id);
    }
    setSwitchingId(id);
    try {
      await api("/api/settings", {
        method: "POST",
        body: JSON.stringify({ enhancer_model: id }),
      });
      await loadModels();
      onSaved?.();
    } catch (e) {
      onFeedback?.({ type: "error", text: e.message || String(e) });
    } finally {
      setSwitchingId(null);
    }
  }

  const active = useMemo(
    () => engines.find((e) => e.key === activeKey) || null,
    [engines, activeKey]
  );

  const draftKey = active ? `${mode}:${active.key}` : "";
  const draft = draftKey ? (drafts[draftKey] ?? "") : "";
  const baseline = active ? draftFor(active, mode) : "";
  const dirty = active ? draft.trim() !== baseline.trim() : false;
  const activeCustom = active ? isCustom(active, mode) : false;
  const modeLabel = mode === "json" ? t("settings.enhancer.modeJson") : t("settings.enhancer.modeText");

  async function save(instructions) {
    if (!active) return;
    setSaving(true);
    try {
      const data = await api(API, {
        method: "POST",
        body: JSON.stringify({ engine_key: active.key, mode, instructions }),
      });
      const list = data.engines || [];
      setEngines(list);
      const updated = list.find((e) => e.key === active.key);
      setDrafts((prev) => ({
        ...prev,
        [`${mode}:${active.key}`]: updated ? draftFor(updated, mode) : "",
      }));
      onFeedback?.({ type: "success", text: t("settings.enhancer.savedFeedback") });
      onSaved?.();
    } catch (e) {
      onFeedback?.({ type: "error", text: e.message || String(e) });
    } finally {
      setSaving(false);
    }
  }

  if (loading) {
    return <p className="hint">{t("settings.enhancer.loading")}</p>;
  }
  if (error) {
    return (
      <p className="error" role="alert">
        {error}
      </p>
    );
  }
  if (!active) {
    return <p className="hint">{t("settings.enhancer.noEngines")}</p>;
  }

  return (
    <div className="enhancer-section">
      <section className="params-section">
        <h3>{t("settings.enhancer.modelTitle")}</h3>
        <p className="params-section-desc">{t("settings.enhancer.modelHint")}</p>
        {modelsLoading ? (
          <p className="hint">{t("settings.enhancer.loading")}</p>
        ) : (
          <div className="enhancer-model-segmented" role="radiogroup" aria-label={t("settings.enhancer.modelTitle")}>
            {[...models].sort((a, b) => ((a.id === "qwen05" ? 0 : 1) - (b.id === "qwen05" ? 0 : 1))).map((m) => (
              <label key={m.id} className={`enhancer-model-card${m.active ? " active" : ""}`}>
                <div className="enhancer-model-card-top">
                  <input
                    type="radio"
                    name="enhancer-model"
                    checked={!!m.active}
                    disabled={switchingId != null || downloadingId != null}
                    onChange={() => activateModel(m.id, m.cached)}
                  />
                  <span className="enhancer-model-name">{m.label}</span>
                  <span className="settings-hint">{m.size}</span>
                </div>
                <div className="enhancer-model-card-bottom">
                  {m.active && (
                    <span className="settings-badge ok">{t("settings.enhancer.activeBadge")}</span>
                  )}
                  {m.cached && !m.active && (
                    <span className="settings-badge">{t("settings.enhancer.downloadedBadge")}</span>
                  )}
                  {!m.cached && (
                    <button
                      type="button"
                      className="btn-mini"
                      disabled={downloadingId != null || switchingId != null}
                      onClick={(e) => {
                        e.preventDefault();
                        downloadModel(m.id);
                      }}
                    >
                      {downloadingId === m.id
                        ? t("settings.enhancer.downloading")
                        : t("settings.enhancer.downloadBtn")}
                    </button>
                  )}
                </div>
              </label>
            ))}
          </div>
        )}
      </section>
      <div className="enhancer-subtabs" role="tablist">
        {engines.map((e) => (
          <button
            key={e.key}
            type="button"
            role="tab"
            aria-selected={e.key === activeKey}
            className={`enhancer-subtab${e.key === activeKey ? " active" : ""}`}
            onClick={() => {
              setActiveKey(e.key);
              setShowPreview(false);
              setShowDefault(false);
            }}
          >
            {e.label.split(" (")[0]}
            {(e.is_custom || e.is_json_custom) && (
              <span className="enhancer-dot" title={t("settings.enhancer.customPromptTitle")} />
            )}
          </button>
        ))}
      </div>

      <div className="enhancer-meta">
        <span className="settings-badge">{active.label}</span>
        <span className="settings-hint">{t("settings.enhancer.targetLength", { length: active.length })}</span>
        {activeCustom ? (
          <span className="settings-badge ok">
            {mode === "json" ? t("settings.enhancer.customJson") : t("settings.enhancer.customText")}
          </span>
        ) : (
          <span className="settings-badge">
            {mode === "json" ? t("settings.enhancer.builtinJson") : t("settings.enhancer.builtinText")}
          </span>
        )}
      </div>

      <div className="enhancer-mode-toggle" role="tablist" aria-label={t("settings.enhancer.modeToggleAria")}>
        <button
          type="button"
          role="tab"
          aria-selected={mode === "text"}
          className={`btn-mini${mode === "text" ? " active" : ""}`}
          onClick={() => {
            setMode("text");
            setShowPreview(false);
          }}
        >
          {t("settings.enhancer.textModeBtn")}
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={mode === "json"}
          className={`btn-mini${mode === "json" ? " active" : ""}`}
          onClick={() => {
            setMode("json");
            setShowPreview(false);
          }}
        >
          {t("settings.enhancer.jsonModeBtn")}
        </button>
      </div>

      <label className="settings-field">
        <span className="settings-field-title">
          {mode === "json"
            ? t("settings.enhancer.jsonFieldTitle")
            : t("settings.enhancer.textFieldTitle")}
        </span>
        <span className="settings-hint">
          {mode === "json"
            ? t("settings.enhancer.jsonFieldHint")
            : t("settings.enhancer.textFieldHint", { model: active.label.split(" (")[0] })}{" "}
          {t("settings.enhancer.modesIndependent")}
        </span>
        <textarea
          className="enhancer-textarea"
          rows={12}
          value={draft}
          onChange={(ev) =>
            setDrafts((prev) => ({ ...prev, [draftKey]: ev.target.value }))
          }
          spellCheck={false}
        />
      </label>

      <div className="settings-actions">
        <button
          type="button"
          className={`btn-mini enhancer-save-btn${dirty ? " dirty" : ""}`}
          disabled={saving || !dirty}
          onClick={() => save(draft)}
          title={dirty ? t("settings.enhancer.saveTitleDirty") : t("settings.enhancer.saveTitleClean")}
        >
          {saving ? t("settings.enhancer.saving") : t("settings.enhancer.saveBtn")}
        </button>
        <button
          type="button"
          className="btn-mini"
          disabled={saving || !activeCustom}
          onClick={() => save("")}
          title={t("settings.enhancer.resetTitle")}
        >
          {t("settings.enhancer.resetBtn")}
        </button>
        <button
          type="button"
          className={`btn-mini${showDefault ? " active" : ""}`}
          onClick={() => setShowDefault((v) => !v)}
        >
          {showDefault ? t("settings.enhancer.hideBuiltin") : t("settings.enhancer.viewBuiltin")}
        </button>
        <button
          type="button"
          className={`btn-mini${showPreview ? " active" : ""}`}
          onClick={() => setShowPreview((v) => !v)}
        >
          {showPreview ? t("settings.enhancer.hidePreview") : t("settings.enhancer.previewFullPrompt")}
        </button>
      </div>

      {showDefault && (
        <div className="enhancer-readonly">
          <span className="settings-field-title">
            {mode === "json"
              ? t("settings.enhancer.builtinJsonTitle")
              : t("settings.enhancer.builtinTextTitle")}
          </span>
          <pre>
            {mode === "json"
              ? active.default_json_instructions
              : active.default_instructions}
          </pre>
        </div>
      )}

      {showPreview && (
        <div className="enhancer-readonly">
          <span className="settings-field-title">
            {t("settings.enhancer.previewTitle", { mode: modeLabel })}
          </span>
          <pre>{mode === "json" ? active.json_system_prompt : active.system_prompt}</pre>
          <span className="settings-hint">
            {mode === "json"
              ? t("settings.enhancer.previewHintJson")
              : t("settings.enhancer.previewHintText")}
          </span>
        </div>
      )}

      <p className="params-hint">{t("settings.enhancer.footerHint")}</p>
    </div>
  );
}