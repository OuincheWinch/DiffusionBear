import { useState } from "react";
import { api } from "../api";
import { useI18n } from "../i18n/I18nContext";

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
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(item.id);
      setTimeout(() => setCopiedId(null), 1500);
    } catch (err) {
      setError("clipboard");
      console.error("[DiffusionBear] clipboard write failed:", err);
    }
  }

  async function handleCopyAllPrompts() {
    const all = items
      .map((it, idx) => `${idx + 1}. [${it.request?.model || "model"}] ${it.request?.prompt || ""}`)
      .join("\n\n");
    if (!all) return;
    try {
      await navigator.clipboard.writeText(all);
      setCopiedAll(true);
      setTimeout(() => setCopiedAll(false), 2000);
    } catch (err) {
      setError("clipboard");
      console.error("[DiffusionBear] clipboard write failed:", err);
    }
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
              🗑 {t("queue.clearAll")}
            </button>
            <button
              type="button"
              className="btn-danger"
              onClick={handleClearAndForget}
              title={t("queue.forgetTitle")}
            >
              🚫 {t("queue.clearAndForget")}
            </button>
          </div>
        )}

        <div className="queue-recovery-list">
          {items.length === 0 ? (
            <div className="queue-recovery-empty">
              <span className="empty-sparkle">✨</span>
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
                      {isInterrupted ? `⚡ ${t("queue.interruptedBadge")}` : `⌫ ${t("queue.cancelledBadge")}`}
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
                      ↺ {t("queue.restore")}
                    </button>
                    <button
                      type="button"
                      className="btn-mini"
                      onClick={() => handleLoadInForm(it)}
                      title={t("queue.loadTitle")}
                    >
                      ✎ {t("queue.load")}
                    </button>
                    <button
                      type="button"
                      className="btn-mini"
                      onClick={() => handleCopyPrompt(it)}
                      title={t("queue.copyTitle")}
                    >
                      {copiedId === it.id ? `✓ ${t("app.copied")}` : `⎘ ${t("app.copy")}`}
                    </button>
                    <button
                      type="button"
                      className="btn-mini btn-item-delete"
                      onClick={() => handleDeleteSingle(it.id)}
                      title={t("queue.deleteOneTitle")}
                    >
                      ✕
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
