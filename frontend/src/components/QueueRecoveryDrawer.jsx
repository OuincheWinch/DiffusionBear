import { useState } from "react";
import { api } from "../api";
import { useI18n } from "../i18n/I18nContext";
import { copyText } from "../clipboard";

export default function QueueRecoveryDrawer({
  isOpen,
  onClose,
  items = [],
  onRefresh,
}) {
  const { t } = useI18n();
  const [restoring, setRestoring] = useState(false);
  const [copiedId, setCopiedId] = useState(null);
  const [copiedAll, setCopiedAll] = useState(false);
  const [error, setError] = useState(null);

  function traceDeleteError(err) {
    // Silent catch blocks hid a real failure for a long time: the buttons looked
    // dead and there was nothing in any log. Keep the detail in the console.
    console.error("[DiffusionBear] recovery delete failed:", err);
  }

  if (!isOpen) return null;

  async function handleRestoreAll() {
    setRestoring(true);
    setError(null);
    try {
      await api("/api/queue/recovery/restore", {
        method: "POST",
        body: JSON.stringify({ job_ids: [] }),
      });
      onRefresh?.();
      onClose();
    } catch (err) {
      setError("restoreFailed");
      console.error("[DiffusionBear] recovery restore failed:", err);
    } finally {
      setRestoring(false);
    }
  }

  async function handleRestoreSingle(id) {
    setRestoring(true);
    setError(null);
    try {
      await api("/api/queue/recovery/restore", {
        method: "POST",
        body: JSON.stringify({ job_ids: [id] }),
      });
      onRefresh?.();
    } catch (err) {
      setError("restoreFailed");
      console.error("[DiffusionBear] recovery restore failed:", err);
    } finally {
      setRestoring(false);
    }
  }

  async function handleDeleteSingle(id) {
    try {
      await api(`/api/queue/recovery?job_id=${encodeURIComponent(id)}`, {
        method: "DELETE",
      });
      onRefresh?.();
    } catch (err) {
      setError("deleteItemFailed");
      traceDeleteError(err);
    }
  }

  async function handleClearAll() {
    if (!window.confirm(t("queue.clearAllConfirm"))) return;
    try {
      await api("/api/queue/recovery", { method: "DELETE" });
      onRefresh?.();
      onClose();
    } catch (err) {
      setError("deleteAllFailed");
      traceDeleteError(err);
    }
  }

  async function handleClearAndForget() {
    if (!window.confirm(t("queue.clearAndForgetConfirm"))) return;
    try {
      await api("/api/queue/recovery?forget=true", { method: "DELETE" });
      onRefresh?.();
      onClose();
    } catch (err) {
      setError("deleteAllFailed");
      traceDeleteError(err);
    }
  }

  function handleLoadInForm(item) {
    const req = item.request || {};
    window.dispatchEvent(
      new CustomEvent("mlx:load-prompt", {
        detail: { ...req },
      })
    );
    onClose();
  }

  async function handleCopyPrompt(item) {
    const text = item.request?.prompt || "";
    if (!text) return;
    // The shared helper adds the legacy fallback; this already reported its failures,
    // which is why it kept working where the gallery's copies did not.
    if (!(await copyText(text))) {
      setError("clipboard");
      return;
    }
    setCopiedId(item.id);
    setTimeout(() => setCopiedId(null), 1500);
  }

  async function handleCopyAllPrompts() {
    const all = items
      .map((it, idx) => `${idx + 1}. [${it.request?.model || "model"}] ${it.request?.prompt || ""}`)
      .join("\n\n");
    if (!all) return;
    if (!(await copyText(all))) {
      setError("clipboard");
      return;
    }
    setCopiedAll(true);
    setTimeout(() => setCopiedAll(false), 2000);
  }

  function formatTime(ts) {
    if (!ts) return "";
    try {
      const d = new Date(ts * 1000);
      return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    } catch {
      return "";
    }
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="queue-recovery-modal"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="queue-recovery-header">
          <div className="queue-recovery-title">
            <span className="queue-recovery-icon">↺</span>
            <h3>{t("queue.recoveryTitle")}</h3>
            <span className="queue-recovery-count-badge">
              {items.length} {items.length > 1 ? t("queue.promptsCount") : t("queue.promptCount")}
            </span>
          </div>
          <button type="button" className="btn-close" onClick={onClose} title={t("app.close")}>
            ✕
          </button>
        </div>

        <p className="queue-recovery-subtitle">{t("queue.recoverySubtitle")}</p>

        {error && (
          <div className="queue-recovery-error" role="alert">
            <span>⚠</span>
            <span>{error === "restoreFailed"
              ? t("queue.restoreFailed")
              : error === "clipboard"
                ? t("queue.clipboardFailed")
                : t("queue.deleteFailed")}</span>
            <button
              type="button"
              className="btn-mini"
              onClick={() => setError(null)}
              title={t("app.dismiss")}
            >
              ✕
            </button>
          </div>
        )}

        {items.length > 0 && (
          <div className="queue-recovery-bulk-actions">
            <button
              type="button"
              className="btn-primary btn-restore-all"
              onClick={handleRestoreAll}
              disabled={restoring}
            >
              ↺ {t("queue.restoreAll")} ({items.length})
            </button>
            <button
              type="button"
              className="btn-secondary"
              onClick={handleCopyAllPrompts}
            >
              {copiedAll ? t("queue.copiedAll") : `⎘ ${t("queue.copyAll")}`}
            </button>
            <button
              type="button"
              className="btn-danger-ghost"
              onClick={handleClearAll}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{ verticalAlign: "-1px", marginRight: "4px" }}>
                <polyline points="3 6 5 6 21 6" />
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
              </svg>
              {t("queue.clearAll")}
            </button>
            <button
              type="button"
              className="btn-danger"
              onClick={handleClearAndForget}
              title={t("queue.forgetTitle")}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{ verticalAlign: "-1px", marginRight: "4px" }}>
                <circle cx="12" cy="12" r="10" />
                <line x1="4.93" y1="4.93" x2="19.07" y2="19.07" />
              </svg>
              {t("queue.clearAndForget")}
            </button>
          </div>
        )}

        <div className="queue-recovery-list">
          {items.length === 0 ? (
            <div className="queue-recovery-empty">
              <span className="empty-sparkle">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z" />
                </svg>
              </span>
              <p>{t("queue.empty")}</p>
              <span className="hint">{t("queue.emptyHint")}</span>
            </div>
          ) : (
            items.map((it) => {
              const req = it.request || {};
              const isInterrupted = it.reason === "interrupted";
              const loraCount = req.loras?.length || 0;

              return (
                <div
                  key={it.id}
                  className={`queue-recovery-item ${isInterrupted ? "interrupted" : "cancelled"}`}
                >
                  <div className="queue-recovery-item-head">
                    <span
                      className={`badge-status ${isInterrupted ? "badge-interrupted" : "badge-cancelled"}`}
                    >
                      {isInterrupted ? t("queue.interruptedBadge") : t("queue.cancelledBadge")}
                    </span>
                    <span className="queue-recovery-item-time">
                      {formatTime(it.timestamp)}
                    </span>
                    <span className="queue-recovery-model-tag">
                      {req.model || t("queue.defaultModel")}
                    </span>
                    {req.width && req.height && (
                      <span className="queue-recovery-spec-pill">
                        {req.width}×{req.height}
                      </span>
                    )}
                    {req.steps && (
                      <span className="queue-recovery-spec-pill">
                        {req.steps} {t("queue.steps")}
                      </span>
                    )}
                    {loraCount > 0 && (
                      <span className="queue-recovery-spec-pill">
                        {loraCount} LoRA{loraCount > 1 ? "s" : ""}
                      </span>
                    )}
                  </div>

                  <div className="queue-recovery-prompt" title={req.prompt}>
                    &ldquo;{req.prompt}&rdquo;
                  </div>

                  <div className="queue-recovery-item-actions">
                    <button
                      type="button"
                      className="btn-mini btn-action-requeue"
                      onClick={() => handleRestoreSingle(it.id)}
                      disabled={restoring}
                      title={t("queue.requeueOneTitle")}
                    >
                      <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{ verticalAlign: "-1px", marginRight: "3px" }}>
                        <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                        <path d="M3 3v5h5" />
                      </svg>
                      {t("queue.restore")}
                    </button>
                    <button
                      type="button"
                      className="btn-mini"
                      onClick={() => handleLoadInForm(it)}
                      title={t("queue.loadTitle")}
                    >
                      <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{ verticalAlign: "-1px", marginRight: "3px" }}>
                        <path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z" />
                      </svg>
                      {t("queue.load")}
                    </button>
                    <button
                      type="button"
                      className="btn-mini"
                      onClick={() => handleCopyPrompt(it)}
                      title={t("queue.copyTitle")}
                    >
                      {copiedId === it.id ? (
                        <>
                          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{ verticalAlign: "-1px", marginRight: "3px" }}>
                            <polyline points="20 6 9 17 4 12" />
                          </svg>
                          {t("app.copied")}
                        </>
                      ) : (
                        <>
                          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{ verticalAlign: "-1px", marginRight: "3px" }}>
                            <rect width="14" height="14" x="8" y="8" rx="2" ry="2" />
                            <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2" />
                          </svg>
                          {t("app.copy")}
                        </>
                      )}
                    </button>
                    <button
                      type="button"
                      className="btn-mini btn-item-delete"
                      onClick={() => handleDeleteSingle(it.id)}
                      title={t("queue.deleteOneTitle")}
                      aria-label={t("queue.deleteOneTitle")}
                    >
                      <svg width="10" height="10" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                        <path d="M2 2l10 10M12 2L2 12" />
                      </svg>
                    </button>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}
