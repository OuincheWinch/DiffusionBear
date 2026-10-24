import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { useI18n } from "../i18n/I18nContext";

export default function PersistentDownloadProgress({
  models = [],
  onInstalled = null,
  onActiveTasksChange = null,
}) {
  const { t } = useI18n();
  const [tasks, setTasks] = useState([]);
  const pollTimerRef = useRef(null);
  const prevDoneIdsRef = useRef(new Set());

  const refreshDownloads = async () => {
    try {
      const allTasks = await api("/api/models/downloads");
      if (Array.isArray(allTasks)) {
        setTasks(allTasks);

        // Check for newly completed tasks to trigger model refresh
        const doneTasks = allTasks.filter((t) => t.status === "done");
        let hasNewDone = false;
        for (const dt of doneTasks) {
          if (!prevDoneIdsRef.current.has(dt.id)) {
            prevDoneIdsRef.current.add(dt.id);
            hasNewDone = true;
          }
        }
        if (hasNewDone) {
          onInstalled?.();
        }

        const active = allTasks.filter((t) =>
          ["downloading", "pending", "copying"].includes(t.status)
        );
        onActiveTasksChange?.(active);
      }
    } catch {
      // Background poll failure, will retry next tick
    }
  };

  useEffect(() => {
    refreshDownloads();
    // Dynamic polling rate: 1000ms while active, 4000ms when idle
    const poll = async () => {
      await refreshDownloads();
      const hasActive = tasks.some((t) =>
        ["downloading", "pending", "copying"].includes(t.status)
      );
      pollTimerRef.current = setTimeout(poll, hasActive ? 1000 : 3500);
    };

    pollTimerRef.current = setTimeout(poll, 1500);
    return () => {
      if (pollTimerRef.current) clearTimeout(pollTimerRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleCancel = async (taskId) => {
    try {
      await api(`/api/models/downloads/${taskId}`, { method: "DELETE" });
      await refreshDownloads();
    } catch (e) {
      console.error("[PersistentDownloadProgress] Failed to cancel download:", e);
    }
  };

  const activeTasks = tasks.filter((t) =>
    ["downloading", "pending", "copying"].includes(t.status)
  );

  if (activeTasks.length === 0) {
    return null;
  }

  return (
    <div className="persistent-downloads-container">
      {activeTasks.map((task) => {
        const modelObj = models.find((m) => m.id === task.model_id);
        const modelLabel = modelObj?.label || task.model_id;
        const totalMb = task.total_bytes ? (task.total_bytes / 1048576).toFixed(0) : null;
        const dlMb = task.downloaded_bytes ? (task.downloaded_bytes / 1048576).toFixed(0) : "0";
        const pct =
          task.total_bytes > 0
            ? Math.min(99, Math.round((task.downloaded_bytes / task.total_bytes) * 100))
            : 0;

        return (
          <div key={task.id} className="persistent-download-card">
            <div className="persistent-download-header">
              <div className="persistent-download-title">
                <svg className="persistent-dl-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                <span className="persistent-dl-name">{modelLabel}</span>
                <span className="persistent-dl-badge">
                  {task.status === "copying"
                    ? t("canvas.progress.phaseSaving") || "Finalizing"
                    : task.status === "pending"
                      ? t("models.preparing") || "Queued"
                      : t("models.downloading") || "Downloading"}
                </span>
              </div>
              <button
                type="button"
                className="btn-cancel-download"
                onClick={() => handleCancel(task.id)}
                title={t("app.cancel")}
                aria-label={t("app.cancel")}
              >
                ✕
              </button>
            </div>

            <div className="civitai-progress-track">
              <div
                className={`civitai-progress-fill${task.status === "done" ? " done" : ""}`}
                style={{ transform: `scaleX(${(task.status === "done" ? 100 : pct) / 100})` }}
              />
            </div>

            <div className="civitai-download-footer">
              <span>
                <strong>{pct}%</strong>
                {task.speed_mb_s > 0 && (
                  <span className="civitai-dl-speed" style={{ marginLeft: "8px" }}>
                    {task.speed_mb_s.toFixed(1)} MB/s
                  </span>
                )}
              </span>
              {totalMb ? (
                <span>
                  {dlMb} / {totalMb} MB
                </span>
              ) : (
                <span>{dlMb} MB</span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
