import { useCallback, useEffect, useState } from "react";
import { api } from "../../api";

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
    if (!window.confirm("Cancel the entire queue? Running generations are stopped and archived for recovery.")) return;
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
    if (!window.confirm("Delete all recovery records? (interrupted prompts stay archived unless you also clear the pending queue).")) return;
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
          {active.length} active · {recovery.length} recoverable
        </span>
        {active.length > 0 && (
          <button type="button" className="btn-mini" disabled={busy} onClick={cancelAll}>
            ⌫ Cancel all
          </button>
        )}
        <button
          type="button"
          className="btn-mini"
          onClick={() => setShowRecovery((v) => !v)}
        >
          {showRecovery ? "Hide recovery" : `↺ Recovery (${recovery.length})`}
        </button>
      </div>

      {active.length === 0 ? (
        <p className="params-hint">
          {jobs === null
            ? "Loading queue…"
            : "Queue idle — no generation currently running or waiting."}
        </p>
      ) : (
        <div className="models-table">
          <div className="models-table-head">
            <span>Prompt / model</span>
            <span>Status</span>
            <span>Queued</span>
            <span>Actions</span>
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
                    : "queued"}
                </span>
              </span>
              <span className="models-size">{fmtTime(j.created_at)}</span>
              <span className="models-actions">
                <button type="button" className="btn-mini" onClick={() => cancelJob(j.id)}>
                  ✕ Cancel
                </button>
              </span>
            </div>
          ))}
        </div>
      )}

      {showRecovery && (
        <div className="recovery-box">
          <div className="settings-inline">
            <strong>↺ Recoverable prompts</strong>
            {recovery.length > 0 && (
              <>
                <button type="button" className="btn-mini" disabled={busy} onClick={() => restoreItem(null)}>
                  ↺ Requeue all ({recovery.length})
                </button>
                <button type="button" className="btn-mini" onClick={clearRecovery}>
                  🗑 Clear history
                </button>
              </>
            )}
          </div>
          {recovery.length === 0 ? (
            <p className="params-hint">
              Nothing recoverable. Cancelled or interrupted generations are archived here
              (queue_recovery.json) and can be requeued after a restart.
            </p>
          ) : (
            <div className="recovery-list">
              {recovery.map((it) => (
                <div className="recovery-item" key={it.id}>
                  <div className="recovery-item-head">
                    <span className="settings-badge">{it.reason === "interrupted" ? "⚡ interrupted" : "⌫ cancelled"}</span>
                    <span className="models-id">{it.request?.model || ""}</span>
                  </div>
                  <div className="recovery-prompt" title={it.request?.prompt}>
                    “{it.request?.prompt}”
                  </div>
                  <div className="recovery-actions">
                    <button type="button" className="btn-mini" disabled={busy} onClick={() => restoreItem(it.id)}>
                      ↺ Requeue
                    </button>
                    <button type="button" className="btn-mini" onClick={() => loadInForm(it)}>
                      ✎ Load in form
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