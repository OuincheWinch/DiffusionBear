import { useState } from "react";
import { api } from "../../api";
import ModelInstaller from "../ModelInstaller";
import { formatBytes } from "../../utils/formatBytes";
import { useI18n } from "../../i18n/I18nContext";
import { useModelIntegrity } from "../../hooks/useModelIntegrity";
import { isUnlinkAction, removeModelEndpoint } from "../../utils/modelActions";
import RamFitBadge from "../RamFitBadge";
import { defaultGenerationSeconds, modelMemoryNeedGb, ramFit, useMachineMemoryGb } from "../../utils/ramFit";

const FAST_SECONDS = 30;
const GROUPS = ["installed", "fast", "quality", "tools"];

function modelGroup(m) {
  if (m.is_utility) return "tools";
  if (m.installed || m.local_path_configured) return "installed";
  const seconds = defaultGenerationSeconds(m);
  return seconds && seconds <= FAST_SECONDS ? "fast" : "quality";
}

function groupModels(models) {
  const byNeed = (a, b) => (modelMemoryNeedGb(a) || 99) - (modelMemoryNeedGb(b) || 99);
  return GROUPS.map((key) => ({
    key,
    models: models.filter((m) => modelGroup(m) === key).sort(byNeed),
  })).filter((g) => g.models.length);
}

export default function ModelsSection({ models, loading, onModelsChanged, onFeedback }) {
  const { t } = useI18n();
  const ramGb = useMachineMemoryGb();
  const [fitsOnly, setFitsOnly] = useState(true);
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
    const isUnlink = isUnlinkAction(m);
    const action = isUnlink ? t("settings.models.confirmUnlink", { model: m.label }) : t("settings.models.confirmRemove", { model: m.label });
    if (!window.confirm(action)) {
      return;
    }
    setRemoving(m.id);
    setErr(null);
    try {
      const endpoint = removeModelEndpoint(m);
      const result = await api(endpoint, { method: "DELETE" });
      await onModelsChanged?.();
      window.dispatchEvent(new CustomEvent("mlx:models-changed"));
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

  const tooLarge = (m) =>
    !m.is_utility && !m.installed && !m.local_path_configured && ramFit(m, ramGb) === "over";
  const hiddenCount = fitsOnly ? models.filter(tooLarge).length : 0;
  const shown = hiddenCount ? models.filter((m) => !tooLarge(m)) : models;

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
      {ramGb > 0 && (
        <div className="model-fit-filter" role="group" aria-label={t("models.filter.label")}>
          {[[false, "all"], [true, "fits"]].map(([value, key]) => (
            <button
              key={key}
              type="button"
              className={`model-fit-filter-btn${fitsOnly === value ? " is-active" : ""}`}
              aria-pressed={fitsOnly === value}
              onClick={() => setFitsOnly(value)}
            >
              {t(`models.filter.${key}`)}
            </button>
          ))}
          {hiddenCount > 0 && (
            <span className="hint model-fit-filter-hidden">{t("models.filter.hidden", { count: hiddenCount })}</span>
          )}
        </div>
      )}
      {groupModels(shown).map((group) => (
        <section key={group.key} className="model-card-group" aria-labelledby={`model-group-${group.key}`}>
          <h4 id={`model-group-${group.key}`} className="model-card-group-title">
            {t(`models.group.${group.key}`)}
            <span className="model-card-group-count">{group.models.length}</span>
          </h4>
          <ul className="model-cards">
            {group.models.map((m) => {
              const installed = !!m.installed;
              const override = !!m.has_model_override;
              const needsUpgrade = !!m.krea_needs_upgrade;
              const isBuiltin = !!m.is_builtin;
              const integrityStatus = m.integrity_status;
              const isRepairingThis = repairing?.modelId === m.id;
              const seconds = defaultGenerationSeconds(m);
              const onDisk = installed || m.local_path_configured;
              return (
                <li className="model-card" key={m.id}>
                  <div className="model-card-head">
                    <strong className="model-card-name" title={m.label}>{m.label}</strong>
                    <RamFitBadge model={m} ramGb={ramGb} />
                  </div>
                  <p className="model-card-meta">
                    {onDisk
                      ? t("models.card.onDisk", { size: formatBytes(m.disk_usage_bytes || 0) })
                      : t("models.card.download", { size: m.size_gb })}
                    {seconds ? ` · ${t("models.card.perImage", { seconds })}` : ""}
                  </p>
                  <div className="models-state">
                    {m.supports_ref && <span className="settings-badge">{t("models.card.tagEdit")}</span>}
                    {m.supports_loras && <span className="settings-badge">LoRA</span>}
                    {isBuiltin && installed && integrityStatus === "ok" && (
                      <span className="settings-badge ok">{t("models.integrity.badgeOk")}</span>
                    )}
                    {needsUpgrade && <span className="settings-badge missing">{t("installer.updateAvailable")}</span>}
                    {override && <span className="settings-badge">{t("settings.models.customDefaults")}</span>}
                  </div>
                  <span className="models-actions model-card-actions">
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
                            title={isUnlinkAction(m) ? t("models.detected.unlinkTitle") : t("settings.models.removeTitle")}
                          >
                            {removing === m.id ? t("settings.models.deleting") : isUnlinkAction(m) ? t("settings.models.unlinkPath") : t("settings.models.remove")}
                          </button>
                       </>
                     ) : (
                       <ModelInstaller modelInfo={m} onInstalled={onModelsChanged} compact />
                     )}
                  </span>
                </li>
              );
            })}
          </ul>
        </section>
      ))}
      <p className="params-hint">
        {t("settings.models.storageHintLead")} <code>backend/data/models/</code> (SDXL),{" "}
        <code>backend/data/models/krea2-turbo-q4/</code>{" "}
        {t("settings.models.storageHintTail")}
      </p>
    </div>
  );
}