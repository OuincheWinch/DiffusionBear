import { useCallback, useEffect, useState, memo } from "react";
import { api } from "../api";
import QueueRecoveryDrawer from "./QueueRecoveryDrawer";
import { useI18n } from "../i18n/I18nContext";

// Maps the backend's status codes to i18n keys rather than to literal text: this
// table lives at module level, outside the component, so it cannot call useI18n().
const STATUS_KEYS = {
  queued: "stack.status.queued",
  generating: "stack.status.generating",
  done: "stack.status.done",
  error: "stack.status.error",
  cancelled: "stack.status.cancelled",
};

function GenerationStack() {
  const { t } = useI18n();
  const [jobs, setJobs] = useState([]);
  const [recoverableItems, setRecoverableItems] = useState([]);
  const [showRecovery, setShowRecovery] = useState(false);

  const fetchRecovery = useCallback(async () => {
    try {
      const res = await api("/api/queue/recovery");
      setRecoverableItems(res.items || []);
    } catch {}
  }, []);

  useEffect(() => {
    let alive = true;
    let timerId = null;

    async function poll() {
      if (timerId) clearTimeout(timerId);
      try {
        const list = await api("/api/jobs?limit=25");
        if (!alive) return;
        const cutoff = Date.now() / 1000 - 120;
        const relevant = list.filter(
          (j) =>
            ["queued", "generating"].includes(j.status) ||
            (j.finished_at ?? 0) > cutoff
        );
        setJobs(relevant);
        fetchRecovery();
        const hasActive = relevant.some((j) =>
          ["queued", "generating"].includes(j.status)
        );
        const isHidden = typeof document !== "undefined" && document.hidden;
        const delay = hasActive ? (isHidden ? 8000 : 4000) : (isHidden ? 30000 : 10000);
        timerId = setTimeout(poll, delay);
      } catch {
        if (alive) {
          const isHidden = typeof document !== "undefined" && document.hidden;
          timerId = setTimeout(poll, isHidden ? 30000 : 10000);
        }
      }
    }

    function handleVisibilityChange() {
      if (typeof document !== "undefined" && !document.hidden && alive) {
        poll();
      }
    }

    if (typeof document !== "undefined") {
      document.addEventListener("visibilitychange", handleVisibilityChange);
    }
    poll();
    fetchRecovery();

    return () => {
      alive = false;
      if (typeof document !== "undefined") {
        document.removeEventListener("visibilitychange", handleVisibilityChange);
      }
      if (timerId) clearTimeout(timerId);
    };
  }, [fetchRecovery]);

  const active = jobs.filter((j) => ["queued", "generating"].includes(j.status));

  // Count only active workload (current and next/queued generations), adjusted by batch number
  const activeImageCount = active.reduce((sum, j) => {
    const totalBatch = j.batch || j.progress?.batch || 1;
    if (j.status === "generating") {
      const currentIndex = j.progress?.image_index ?? 0;
      const remainingInBatch = Math.max(1, totalBatch - currentIndex);
      return sum + remainingInBatch;
    }
    return sum + totalBatch;
  }, 0);

  async function cancelJob(id) {
    window.dispatchEvent(new CustomEvent("mlx:cancel-job", { detail: { id } }));
    try {
      await api(`/api/jobs/${id}/cancel`, { method: "POST" });
      setJobs((prev) =>
        prev.map((j) => (j.id === id ? { ...j, status: "cancelled" } : j))
      );
      fetchRecovery();
    } catch {}
  }

  async function emptyQueue() {
    window.dispatchEvent(new CustomEvent("mlx:cancel-all"));
    try {
      await api("/api/jobs/cancel-all", { method: "POST" });
      setJobs((prev) =>
        prev.map((j) =>
          ["queued", "generating"].includes(j.status)
            ? { ...j, status: "cancelled" }
            : j
        )
      );
      fetchRecovery();
    } catch {}
  }

  const interruptedCount = recoverableItems.filter((x) => x.reason === "interrupted").length;

  if (jobs.length === 0 && recoverableItems.length === 0) return null;

  return (
    <>
      <div className="gen-stack">
        <h3>
          {t("stack.heading", { count: activeImageCount })}
          <span className="gen-stack-actions">
            {active.some((j) => j.status === "generating") && (
              <button
                type="button"
                className="btn-mini"
                onClick={() => {
                  const run = active.find((j) => j.status === "generating");
                  if (run) cancelJob(run.id);
                }}
              >
                {t("stack.killCurrentBtn")}
              </button>
            )}
            {active.length > 1 && (
              <button type="button" className="btn-mini" onClick={emptyQueue}>
                {t("stack.emptyQueueBtn")}
              </button>
            )}
            {recoverableItems.length > 0 && (
              <button
                type="button"
                className="btn-mini btn-recovery-trigger"
                onClick={() => setShowRecovery(true)}
                title={t("stack.recoveryTriggerTitle")}
              >
                {t(
                  recoverableItems.length > 1 ? "stack.recoveryTriggerMany" : "stack.recoveryTriggerOne",
                  { count: recoverableItems.length }
                )}
              </button>
            )}
          </span>
        </h3>

        {interruptedCount > 0 && (
          <div className="queue-crash-alert">
            <span>
              {t(
                interruptedCount > 1 ? "stack.crashAlertMany" : "stack.crashAlertOne",
                { count: interruptedCount }
              )}
            </span>
            <button
              type="button"
              className="btn-mini btn-crash-restore"
              onClick={() => setShowRecovery(true)}
            >
              {t("stack.crashAlertBtn")}
            </button>
          </div>
        )}

        {jobs.map((j) => {
          const p = j.progress;
          const pct =
            j.status === "done"
              ? 100
              : p && p.steps > 0
                ? Math.min(100, (p.step / p.steps) * 100)
                : null;
          return (
            <div key={j.id} className={`gen-stack-item ${j.status}`}>
              <div className="gen-stack-head">
                <span className={`gen-stack-status ${j.phase ? `phase-${j.phase}` : ""}`}>
                  {j.status === "generating" && j.phase === "downloading"
                    ? t("stack.phase.downloading")
                    : j.status === "generating" && j.phase === "loading_model"
                      ? t("stack.phase.loadingMemory")
                      : j.status === "generating" && j.phase === "compiling"
                        ? t("stack.phase.compiling")
                        : j.status === "generating" && j.phase === "saving"
                          ? t("stack.phase.finalizing")
                          : STATUS_KEYS[j.status] ? t(STATUS_KEYS[j.status]) : j.status}
                </span>
                <span className="gen-stack-model">{j.model}</span>
              </div>
              {["queued", "generating"].includes(j.status) && (
                <button
                  type="button"
                  className="btn-mini gen-stack-kill"
                  title={t("stack.job.cancelTitle")}
                  onClick={() => cancelJob(j.id)}
                >
                  ✕
                </button>
              )}
              <div className="gen-stack-prompt" title={j.prompt}>
                {j.prompt}
              </div>
              {/* The reason, when there is one.

                  This row used to say only "X Error" with no explanation, while the banner
                  at the top of the form carried the real Metal message. Two surfaces
                  describing the same failure differently read as two separate errors -- and
                  the stack is what you look at after dismissing the banner, so it was the
                  worse of the two: no reason at all. */}
              {j.status === "error" && j.error && (
                <div className="gen-stack-error" title={j.error}>
                  {j.error}
                </div>
              )}
              {pct != null && (
                <div className="progress-bar-hairline">
                  <div
                    className="progress-fill"
                    // scaleX rather than width: animating width re-runs layout every tick.
                    style={{ transform: `scaleX(${pct / 100})` }}
                  />
                </div>
              )}
              {p ? (
                <div className="hint">
                  {j.status === "done" ? (
                    <>
                      {t("stack.job.completedIn", {
                        time: j.result?.generation_time ?? Math.round(p.elapsed),
                        steps: p.steps,
                      })}
                      {(p.batch > 1 || j.batch > 1)
                        ? t("stack.job.completedBatch", { batch: p.batch || j.batch })
                        : null}
                    </>
                  ) : (
                    <>
                      {p.step >= p.steps ? (
                        t("stack.job.decoding", { elapsed: Math.round(p.elapsed) })
                      ) : (
                        t("stack.job.stepElapsed", {
                          step: p.step,
                          steps: p.steps,
                          elapsed: Math.round(p.elapsed),
                        })
                      )}
                      {p.eta_seconds != null && p.eta_seconds >= 0 && (
                        t("stack.job.eta", { eta: Math.round(p.eta_seconds) })
                      )}
                      {(p.batch > 1 || j.batch > 1) && (
                        t("stack.job.imageProgress", {
                          index: (p.image_index ?? 0) + 1,
                          batch: p.batch || j.batch,
                        })
                      )}
                    </>
                  )}
                </div>
              ) : j.status === "queued" ? (
                <div className="hint">
                  {t("stack.job.waitingInQueue")}
                  {(j.batch > 1) ? t("stack.job.waitingBatch", { batch: j.batch }) : null}
                </div>
              ) : j.status === "generating" && !p ? (
                <div className="hint">
                  {j.phase_detail || (j.phase === "downloading"
                    ? t("stack.job.phaseDownloading")
                    : t("stack.job.phaseLoadingWeights"))}
                </div>
              ) : null}
            </div>
          );
        })}
      </div>

      <QueueRecoveryDrawer
        isOpen={showRecovery}
        onClose={() => setShowRecovery(false)}
        items={recoverableItems}
        onRefresh={fetchRecovery}
      />
    </>
  );
}

export default memo(GenerationStack);
