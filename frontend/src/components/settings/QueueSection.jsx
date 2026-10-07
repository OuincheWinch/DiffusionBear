import { useCallback, useEffect, useState } from "react";
import { api } from "../../api";
import { useI18n } from "../../i18n/I18nContext";

const STATUS_CLASS = {
  generating: "badge-generating",
  queued: "badge-queued",
  done: "badge-done",
  error: "badge-error",
  cancelled: "badge-cancelled",
};

function fmtTime(ts) {
  if (!ts) return "";
  try {
    return new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch {
    return "";
  }
}

export default function QueueSection({ onNavigate }) {
  const { t } = useI18n();
  const [jobs, setJobs] = useState(null);
  const [active, setActive] = useState([]);
  const [recovery, setRecovery] = useState([]);
  const [showRecovery, setShowRecovery] = useState(false);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const jobsData = await api("/api/jobs?limit=12");
      setJobs(jobsData.filter((j) => j.status === "generating" || j.status === "queued"));
      const rec = await api("/api/queue/recovery");
      setRecovery(rec.items || []);
      setActive(jobsData.filter((j) => j.status === "generating" || j.status === "queued"));
    } catch {}
  }, []);

  useEffect(() => {
    refresh();
    let timer = null;
    const tick = () => {
      const hidden = typeof document !== "undefined" && document.hidden;
      timer = setTimeout(async () => {
        await refresh();
        tick();
      }, hidden ? 20000 : 4000);
    };
    tick();
    const onVisibility = () => {
      if (typeof document !== "undefined" && !document.hidden) {
        clearTimeout(timer);
        tick();
      }
    };
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      document.removeEventListener("visibilitychange", onVisibility);
      clearTimeout(timer);
    };
  }, [refresh]);

  async function cancelJob(id) {
    try {
      await api(`/api/jobs/${id}/cancel`, { method: "POST" });
      await refresh();
    } catch (e) {
      alert(String(e.message || e));
    }
  }

  async function cancelAll() {
    if (!window.confirm(t("settings.queue.cancelAllConfirm"))) return;
    setBusy(true);
    try {
      await api("/api/jobs/cancel-all", { method: "POST" });
      await refresh();
    } finally {
      setBusy(false);
    }
  }

  async function restoreItem(id) {
    setBusy(true);
    try {
      await api("/api/queue/recovery/restore", {
        method: "POST",
        body: JSON.stringify({ job_ids: id ? [id] : [] }),
      });
      await refresh();
    } finally {
      setBusy(false);
    }
  }

  async function deleteRecoveryItem(id) {
    try {
      await api(`/api/queue/recovery?job_id=${encodeURIComponent(id)}`, { method: "DELETE" });
      await refresh();
    } catch {}
  }

  async function clearRecovery() {
    if (!window.confirm(t("settings.queue.clearRecoveryConfirm"))) return;
    try {
      await api("/api/queue/recovery", { method: "DELETE" });
      await refresh();
    } catch {}
  }

  function loadInForm(item) {
    const req = item.request || {};
    window.dispatchEvent(new CustomEvent("mlx:load-prompt", { detail: req }));
    onNavigate?.("generate");
  }

  const live = active.length > 0;

  return (
    <div className="settings-row">
      <div className="settings-inline">
        <span className={`settings-badge ${live ? "ok" : ""}`}>
          {t("settings.queue.counts", { active: active.length, recovery: recovery.length })}
        </span>
        {/* Both destructive actions live HERE, in the row that is always visible.
         *
         * They used to be reachable only by expanding Recovery, and Cancel all only
         * appeared when something was running. So in the most common state -- 0 active
         * and a few recoverable prompts left over from a crash -- the panel offered no way
         * to delete anything at all: you had to expand a list just to find the button
         * that clears it. Nothing is lost by expanding a list to read it; losing the
         * ability to delete without a pointless detour is a real defect. */}
        {active.length > 0 && (
          <button type="button" className="btn-mini" disabled={busy} onClick={cancelAll}>
            {t("settings.queue.cancelAll")}
          </button>
        )}
        {recovery.length > 0 && (
          <button
            type="button"
            className="btn-mini danger"
            disabled={busy}
            onClick={clearRecovery}
            title={t("settings.queue.clearRecoveryConfirm")}
          >
            {t("settings.queue.clearHistory")}
          </button>
        )}
        <button
          type="button"
          className="btn-mini"
          onClick={() => setShowRecovery((v) => !v)}
        >
          {showRecovery ? t("settings.queue.hideRecovery") : t("settings.queue.recovery", { count: recovery.length })}
        </button>
      </div>

      {active.length === 0 ? (
        <p className="params-hint">
          {jobs === null
            ? t("settings.queue.loading")
            : t("settings.queue.idle")}
        </p>
      ) : (
        <div className="models-table">
          <div className="models-table-head">
            <span>{t("settings.queue.colPromptModel")}</span>
            <span>{t("settings.queue.colStatus")}</span>
            <span>{t("settings.queue.colQueued")}</span>
            <span>{t("settings.queue.colActions")}</span>
          </div>
          {active.map((j) => (
            <div className="models-table-row" key={j.id}>
              <span className="models-name">
                <strong>“{j.prompt}”</strong>
                <span className="models-id">{j.model}</span>
              </span>
              <span className="models-state">
                <span className={`settings-badge ${STATUS_CLASS[j.status] || ""}`}>
                  {j.status === "generating"
                    ? `⚙ ${j.progress?.step ?? "…"}/${j.progress?.steps ?? "…"}`
                    : t("settings.queue.queued")}
                </span>
              </span>
              <span className="models-size">{fmtTime(j.created_at)}</span>
              <span className="models-actions">
                <button type="button" className="btn-mini" onClick={() => cancelJob(j.id)}>
                  {t("settings.queue.cancel")}
                </button>
              </span>
            </div>
          ))}
        </div>
      )}

      {showRecovery && (
        <div className="recovery-box">
          <div className="settings-inline">
            <strong>{t("settings.queue.recoverableTitle")}</strong>
            {recovery.length > 0 && (
              <>
                <button type="button" className="btn-mini" disabled={busy} onClick={() => restoreItem(null)}>
                  {t("settings.queue.requeueAll", { count: recovery.length })}
                </button>
              </>
            )}
          </div>
          {recovery.length === 0 ? (
            <p className="params-hint">{t("settings.queue.nothingRecoverable")}</p>
          ) : (
            <div className="recovery-list">
              {recovery.map((it) => (
                <div className="recovery-item" key={it.id}>
                  <div className="recovery-item-head">
                    <span className="settings-badge">{it.reason === "interrupted" ? t("settings.queue.interrupted") : t("settings.queue.cancelled")}</span>
                    <span className="models-id">{it.request?.model || ""}</span>
                  </div>
                  <div className="recovery-prompt" title={it.request?.prompt}>
                    “{it.request?.prompt}”
                  </div>
                  <div className="recovery-actions">
                    <button type="button" className="btn-mini" disabled={busy} onClick={() => restoreItem(it.id)}>
                      {t("settings.queue.requeue")}
                    </button>
                    <button type="button" className="btn-mini" onClick={() => loadInForm(it)}>
                      {t("settings.queue.loadInForm")}
                    </button>
                    <button type="button" className="btn-mini" onClick={() => deleteRecoveryItem(it.id)}>
                      ✕
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}