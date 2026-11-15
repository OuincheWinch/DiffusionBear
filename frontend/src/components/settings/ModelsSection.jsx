import { useState } from "react";
import { api } from "../../api";
import ModelInstaller from "../ModelInstaller";
import { formatBytes } from "../../utils/formatBytes";
import { useI18n } from "../../i18n/I18nContext";
import { useModelIntegrity } from "../../hooks/useModelIntegrity";

export default function ModelsSection({ models, loading, onModelsChanged, onFeedback }) {
  const { t } = useI18n();
  const [removing, setRemoving] = useState(null);
  const [err, setErr] = useState(null);
  const {
    integrity,
    repairing,
    actionError,
    verifyAll,
    verifyOne,
    repairModel,
  } = useModelIntegrity({ onVerificationComplete: onModelsChanged });

  async function handleRepair(m, mode) {
    setErr(null);
    const res = await repairModel(m.id, mode);
    if (res?.status === "already_ok") {
      onFeedback?.({ type: "success", text: t("models.integrity.alreadyOk", { model: m.label }) });
      await onModelsChanged?.();
    } else if (res?.task_id) {
      onFeedback?.({
        type: "success",
        text: t(mode === "full" ? "models.integrity.fullStarted" : "models.integrity.repairStarted", { model: m.label }),
      });
    }
  }

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

  // Models tab stays silent when built-in models need repair; errors surface on Generate.
  const hasAnyIssue = models.some(
    (m) => m.is_builtin && m.integrity_status && m.integrity_status !== "ok"
  );

  return (
    <div className="settings-row">
      {(err || actionError) && <p className="error" role="alert">{err || actionError}</p>}
      <div className="integrity-toolbar">
        <div className="integrity-toolbar-status">
          {integrity?.running ? (
            <>
              <span className="integrity-spinner" aria-hidden="true" />
              <span>
                {t("models.integrity.hashingBanner", {
                  version: integrity.app_version,
                  model: integrity.current_label || "...",
                  done: Math.min((integrity.completed_models || 0) + 1, integrity.total_models || 1),
                  total: integrity.total_models || 1,
                })}
              </span>
            </>
          ) : !hasAnyIssue && integrity?.verified_app_version ? (
            <span className="hint">
              {t("models.integrity.verifiedSummary", { version: integrity.verified_app_version })}
            </span>
          ) : null}
        </div>
        <button
          type="button"
          className="btn-mini"
          disabled={Boolean(integrity?.running)}
          onClick={() => verifyAll()}
        >
          {integrity?.running ? t("models.integrity.verifying") : t("models.integrity.verifyAll")}
        </button>
      </div>
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
          const needsUpgrade = !!m.krea_needs_upgrade;
          const isBuiltin = !!m.is_builtin;
          const integrityStatus = m.integrity_status;
          const isRepairingThis = repairing?.modelId === m.id;
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
                {isBuiltin && installed && integrityStatus === "ok" && (
                  <span className="settings-badge ok">{t("models.integrity.badgeOk")}</span>
                )}
                {needsUpgrade && <span className="settings-badge missing">{t("installer.updateAvailable")}</span>}
                {override && <span className="settings-badge">{t("settings.models.customDefaults")}</span>}
              </span>
              <span className="models-size">{formatBytes(m.disk_usage_bytes || 0)}</span>
              <span className="models-actions">
                 {(installed || m.local_path_configured) && !needsUpgrade ? (
                   <>
                     {isBuiltin && (
                       <>
                         <button
                           type="button"
                           className="btn-mini"
                           disabled={isRepairingThis || Boolean(integrity?.running)}
                           onClick={() => handleRepair(m, "partial")}
                           title={t("models.integrity.repairPartialTitle")}
                         >
                           {t("models.integrity.repair")}
                         </button>
                         <button
                           type="button"
                           className="btn-mini"
                           disabled={isRepairingThis || Boolean(integrity?.running)}
                           onClick={() => handleRepair(m, "full")}
                           title={t("models.integrity.repairFullTitle")}
                         >
                           {t("models.integrity.fullRedownload")}
                         </button>
                         <button
                           type="button"
                           className="btn-mini"
                           disabled={Boolean(integrity?.running)}
                           onClick={() => verifyOne(m.id)}
                         >
                           {t("models.integrity.verifyOne")}
                         </button>
                       </>
                     )}
                     <button
                       type="button"
                       className="btn-mini"
                       disabled={removing === m.id}
                        onClick={() => removeModel(m)}
                        title={t("settings.models.removeTitle")}
                      >
                        {removing === m.id ? t("settings.models.deleting") : m.local_path_configured ? t("settings.models.unlinkPath") : t("settings.models.remove")}
                      </button>
                   </>
                 ) : (
                   <ModelInstaller modelInfo={m} onInstalled={onModelsChanged} compact />
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