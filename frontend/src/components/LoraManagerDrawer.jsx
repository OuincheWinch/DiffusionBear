import { useState } from "react";
import { api } from "../api";
import { findLoraEntry, isKreaDistillLora } from "../utils/loraUtils";
import { useI18n } from "../i18n/I18nContext";

export default function LoraManagerDrawer({
  loras,
  setLoras,
  loraRegistry,
  setLoraRegistry,
  modelInfo,
  engineBase,
  compatibleLoras,
  onSwitchToLoraModel,
  onFeedback,
  children,
}) {
  const { t } = useI18n();
  const [showAllLoras, setShowAllLoras] = useState(false);
  const [syncingCivitai, setSyncingCivitai] = useState(false);

  async function handleSyncCivitai() {
    setSyncingCivitai(true);
    onFeedback?.({ type: "info", text: t("lora.syncingInfo") });
    try {
      const res = await api("/api/loras/sync-civitai", { method: "POST" });
      onFeedback?.({
        type: "success",
        text: t("lora.syncComplete", { updated: res.updated || 0, total: res.total || 0 }),
      });
      const updated = await api("/api/loras");
      setLoraRegistry(updated);
    } catch (e) {
      onFeedback?.({ type: "error", text: t("lora.syncFailed", { message: e.message }) });
    } finally {
      setSyncingCivitai(false);
    }
  }

  function addLoraFromRegistry(name) {
    if (loras.length >= 16) {
      onFeedback?.({ type: "error", text: t("lora.maxActive") });
      return;
    }
    const reg = loraRegistry.find((r) => r.name === name);
    if (!reg) return;
    setLoras([
      ...loras,
      {
        path: reg.path,
        scale: reg.scale ?? 1.0,
        name: reg.name,
        base_model: reg.base_model,
        civitai_version_id: reg.civitai_version_id,
        civitai_model_id: reg.civitai_model_id,
        civitai_model_name: reg.civitai_model_name,
      },
    ]);
  }

  function handleRemoveLora(index) {
    setLoras(loras.filter((_, i) => i !== index));
  }

  async function handleDeleteRegisteredLora(name) {
    const reg = loraRegistry.find((r) => r.name === name);
    if (!reg) return;
    if (!window.confirm(t("lora.deleteConfirm", { name }))) {
      return;
    }
    try {
      await api(`/api/loras/${encodeURIComponent(name)}`, { method: "DELETE" });
      setLoraRegistry((prev) => prev.filter((r) => r.name !== name));
      setLoras((prev) => prev.filter((l) => l.path !== reg.path && l.name !== name));
      onFeedback?.({ type: "info", text: t("lora.deleteSuccess", { name }) });
    } catch (e) {
      onFeedback?.({ type: "error", text: t("lora.deleteFailed", { message: e.message }) });
    }
  }

  if (!modelInfo.lora_format && !modelInfo.supports_loras) {
    return null;
  }

  return (
    <fieldset className="lora-section">
      <legend className="lora-legend">
        <span>{t("lora.activeLegend", { count: loras.length })}</span>
        <button
          type="button"
          className="btn-sync-civitai"
          onClick={handleSyncCivitai}
          disabled={syncingCivitai}
          title={t("lora.syncTitle")}
        >
          {syncingCivitai ? t("lora.syncing") : t("lora.syncBtn")}
        </button>
      </legend>

      {loras.map((l, i) => {
        const regEntry = findLoraEntry(l, loraRegistry);
        const civitaiId = regEntry?.civitai_version_id || l.civitai_version_id;
        const civitaiModelId = regEntry?.civitai_model_id || l.civitai_model_id;
        const displayName =
          regEntry?.civitai_model_name || regEntry?.name || l.name || l.path.split("/").pop();
        const isDistill = isKreaDistillLora(l) || isKreaDistillLora(regEntry);

        return (
          <div className="lora-chip" key={l.path || i}>
            <div className="lora-chip-info">
              <span className="lora-chip-name" title={l.path}>
                {displayName}
              </span>
              {isDistill && (
                <span className="distill-tag" title={t("lora.distillTitle")}>
                  {t("lora.distillTag")}
                </span>
              )}
              {civitaiId && (
                <a
                  href={`https://civitai.red/models/${civitaiModelId || ""}?modelVersionId=${civitaiId}&ref_code=88C8VEBA`}
                  target="_blank"
                  rel="noreferrer"
                  className="civitai-badge"
                  title={t("lora.civitaiIdTitle", { id: civitaiId, version: regEntry?.civitai_version_name || "" })}
                >
                  #{civitaiId} ↗
                </a>
              )}
            </div>
            <input
              type="range"
              min="0"
              max="10"
              step="0.05"
              value={l.scale}
              onChange={(e) => {
                const next = [...loras];
                next[i] = { ...l, scale: Number(e.target.value) };
                setLoras(next);
              }}
            />
            <span className="lora-scale">
              <button
                type="button"
                className="btn-mini"
                title={t("lora.scaleDecreaseTitle")}
                onClick={() => {
                  const next = [...loras];
                  const cur = Number(l.scale ?? 1.0);
                  next[i] = { ...l, scale: Math.max(0, Number((cur - 0.05).toFixed(2))) };
                  setLoras(next);
                }}
              >
                −
              </button>
              {(Number(l.scale) || 1.0).toFixed(2)}
              <button
                type="button"
                className="btn-mini"
                title={t("lora.scaleIncreaseTitle")}
                onClick={() => {
                  const next = [...loras];
                  const cur = Number(l.scale ?? 1.0);
                  next[i] = { ...l, scale: Math.min(10, Number((cur + 0.05).toFixed(2))) };
                  setLoras(next);
                }}
              >
                +
              </button>
            </span>
            <button
              type="button"
              className="lora-remove-btn"
              onClick={() => handleRemoveLora(i)}
            >
              ✕
            </button>
          </div>
        );
      })}

      <select
        value=""
        onChange={(e) => e.target.value && addLoraFromRegistry(e.target.value)}
        disabled={compatibleLoras.length === 0}
      >
        <option value="">
          {compatibleLoras.length === 0
            ? t("lora.registryEmpty", { format: modelInfo.lora_format })
            : t("lora.addOption")}
        </option>
        {compatibleLoras.map((r) => (
          <option key={r.name} value={r.name}>
            {r.name} {r.civitai_version_id ? t("lora.civitaiTagSuffix", { id: r.civitai_version_id }) : ""}
          </option>
        ))}
      </select>

      {compatibleLoras.length > 0 && (
        <div className="lora-registry-chips">
          <span className="lora-registry-title">{t("lora.installedPrefix")}</span>
          {compatibleLoras.map((r) => (
            <span key={r.name} className="lora-chip">
              <span
                className="lora-chip-name"
                title={t("lora.addActiveTitle")}
                onClick={() => addLoraFromRegistry(r.name)}
              >
                {r.name}
              </span>
              <button
                type="button"
                className="lora-delete-chip-btn"
                title={t("lora.deleteChipTitle", { name: r.name })}
                onClick={(e) => {
                  e.stopPropagation();
                  handleDeleteRegisteredLora(r.name);
                }}
              >
                🗑️
              </button>
            </span>
          ))}
        </div>
      )}

      <div className="lora-hub-container">
        <button
          type="button"
          className="btn-toggle-hub"
          onClick={() => setShowAllLoras((v) => !v)}
        >
          {showAllLoras
            ? t("lora.hideHub")
            : t("lora.showHub", { count: loraRegistry.length })}
        </button>
        {showAllLoras && (
          <div className="lora-hub-drawer">
            {loraRegistry.length === 0 ? (
              <p className="hint">{t("lora.hubEmpty")}</p>
            ) : (
              <div className="lora-hub-list">
                {loraRegistry.map((r) => {
                  const isCompatible = r.base_model === engineBase;
                  const isActive = loras.some((l) => l.name === r.name || l.path === r.path);
                  return (
                    <div
                      key={r.name}
                      className={`lora-hub-card ${isCompatible ? "compatible" : "incompatible"}`}
                    >
                      <div className="lora-hub-card-left">
                        <span className={`lora-badge lora-badge-${r.base_model}`}>
                          {r.base_model?.toUpperCase() || "SDXL"}
                        </span>
                        <span className="lora-hub-name" title={r.path}>
                          {r.name}
                        </span>
                        {r.civitai_version_id && (
                          <span className="lora-hub-civitai-tag">#{r.civitai_version_id}</span>
                        )}
                      </div>
                      <div className="lora-hub-card-actions">
                        {isCompatible ? (
                          <button
                            type="button"
                            className={`btn-hub-add ${isActive ? "active" : ""}`}
                            onClick={() => {
                              if (!isActive) addLoraFromRegistry(r.name);
                            }}
                            disabled={isActive}
                          >
                            {isActive ? t("lora.hubActive") : t("lora.hubAdd")}
                          </button>
                        ) : (
                          <button
                            type="button"
                            className="btn-hub-switch"
                            onClick={() => onSwitchToLoraModel?.(r.base_model, r)}
                            title={t("lora.switchTitle", { model: r.base_model?.toUpperCase() })}
                          >
                            {t("lora.switchBtn", { model: r.base_model?.toUpperCase() })}
                          </button>
                        )}
                        <button
                          type="button"
                          className="lora-delete-chip-btn"
                          title={t("lora.deleteChipTitle", { name: r.name })}
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDeleteRegisteredLora(r.name);
                          }}
                        >
                          🗑️
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}
      </div>
      {children}
    </fieldset>
  );
}
