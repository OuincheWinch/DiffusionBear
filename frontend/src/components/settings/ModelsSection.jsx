import { useState } from "react";
import { api } from "../../api";
import ModelInstaller from "../ModelInstaller";
import { formatBytes } from "../../utils/formatBytes";
import { useI18n } from "../../i18n/I18nContext";

export default function ModelsSection({ models, loading, onModelsChanged, onFeedback }) {
  const { t } = useI18n();
  const [removing, setRemoving] = useState(null);
  const [err, setErr] = useState(null);

  async function removeModel(m) {
    const action = m.local_path || m.local_path_configured ? t("settings.models.confirmUnlink", { model: m.label }) : t("settings.models.confirmRemove", { model: m.label });
    if (!window.confirm(action)) {
      return;
    }
    setRemoving(m.id);
    setErr(null);
    try {
      const result = await api(`/api/models/${m.id}`, { method: "DELETE" });
      await onModelsChanged?.();
      onFeedback?.({ type: "success", text: result?.status === "unlinked" ? t("settings.models.feedbackUnlinked", { model: m.label }) : t("settings.models.feedbackRemoved", { model: m.label }) });
    } catch (e) {
      setErr(e.message || String(e));
    } finally {
      setRemoving(null);
    }
  }

  if (loading) {
    return <div className="settings-row"><span className="hint">{t("settings.models.loading")}</span></div>;
  }

  return (
    <div className="settings-row">
      {err && <p className="error" role="alert">{err}</p>}
      <div className="models-table">
        <div className="models-table-head">
          <span>{t("settings.models.colModel")}</span>
          <span>{t("settings.models.colState")}</span>
          <span>{t("settings.models.colOnDisk")}</span>
          <span>{t("settings.models.colActions")}</span>
        </div>
        {models.map((m) => {
          const installed = !!m.installed;
          const override = !!m.has_model_override;
          return (
            <div className="models-table-row" key={m.id}>
              <span className="models-name">
                <strong>{m.label}</strong>
                <span className="models-id">{m.id}</span>
              </span>
              <span className="models-state">
                <span className={`settings-badge ${installed ? "ok" : "missing"}`}>
                  {installed ? t("settings.models.installed") : t("settings.models.notInstalled")}
                </span>
                {override && <span className="settings-badge">{t("settings.models.customDefaults")}</span>}
              </span>
              <span className="models-size">{formatBytes(m.disk_usage_bytes || 0)}</span>
              <span className="models-actions">
                 {installed || m.local_path_configured ? (
                   <button
                     type="button"
                     className="btn-mini"
                     disabled={removing === m.id}
                      onClick={() => removeModel(m)}
                      title={t("settings.models.removeTitle")}
                    >
                      {removing === m.id ? t("settings.models.deleting") : m.local_path_configured ? t("settings.models.unlinkPath") : t("settings.models.remove")}
                    </button>
                 ) : (
                   <ModelInstaller modelInfo={m} onInstalled={onModelsChanged} />
                 )}
              </span>
            </div>
          );
        })}
      </div>
      <p className="params-hint">
        {t("settings.models.storageHintLead")} <code>backend/data/models/</code> (SDXL),{" "}
        <code>backend/data/models/krea2-turbo-q4/</code>{" "}
        {t("settings.models.storageHintTail")}
      </p>
    </div>
  );
}