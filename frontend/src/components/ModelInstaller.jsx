import { useEffect, useRef, useState } from "react";
import { api, pickPath } from "../api";
import { civitaiModelUrl } from "../civitai";
import { useI18n } from "../i18n/I18nContext";

const mb = (bytes) => `${(bytes / 1048576).toFixed(0)} MB`;

export default function ModelInstaller({ modelInfo, onInstalled, compact = false }) {
  const { t } = useI18n();
  const [task, setTask] = useState(null);
  const [starting, setStarting] = useState(false);
  const [showLocal, setShowLocal] = useState(false);
  const [sources, setSources] = useState([]);
  const [sourcesLoading, setSourcesLoading] = useState(false);
  const [picked, setPicked] = useState("");
  const [manualPath, setManualPath] = useState("");
  const [installing, setInstalling] = useState(false);
  const [localError, setLocalError] = useState(null);
  const notifiedDoneRef = useRef(new Set());

  const handleBrowse = async () => {
    try {
      const res = await pickPath({ kind: "folder", title: t("installer.pathPlaceholder") });
      if (!res.cancelled && res.path) {
        setManualPath(res.path);
        setPicked("");
        setLocalError(null);
      }
    } catch (e) {
      setLocalError(e.message || String(e));
    }
  };

  const refresh = async () => {
    try {
      const tasks = await api("/api/models/downloads");
      const matching = tasks.filter((t) => t.model_id === modelInfo.id);
      const mine = matching.find((t) => t.status === "downloading" || t.status === "pending") || matching[0] || null;
      setTask(mine);
      const done = tasks.find((t) => t.model_id === modelInfo.id && t.status === "done");
      if (done && !notifiedDoneRef.current.has(done.id)) {
        notifiedDoneRef.current.add(done.id);
        onInstalled?.();
      }
      return mine;
    } catch (e) {
      setLocalError(e.message || String(e));
      return null;
    }
  };

  useEffect(() => {
    let cancelled = false;
    let timer = null;
    const poll = async () => {
      const next = await refresh();
      if (!cancelled && next && (next.status === "pending" || next.status === "downloading")) {
        timer = setTimeout(poll, 1000);
      }
    };
    poll();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelInfo?.id, task?.status]);

  const start = async () => {
    setStarting(true);
    setLocalError(null);
    try {
      const res = await api("/api/models/download", {
        method: "POST",
        body: JSON.stringify({ model_id: modelInfo.id }),
      });
      if (res?.status === "already_installed") {
        await onInstalled?.();
      } else {
        await refresh();
      }
    } catch (e) {
      setLocalError(e.message || String(e));
    } finally {
      setStarting(false);
    }
  };

  const cancel = async () => {
    if (task) {
      await api(`/api/models/downloads/${task.id}`, { method: "DELETE" });
      await refresh();
    }
  };

  const openLocal = async () => {
    setShowLocal(true);
    setLocalError(null);
    setSourcesLoading(true);
    try {
      const res = await api(
        `/api/models/local-sources?model_id=${encodeURIComponent(modelInfo.id)}`
      );
      setSources(res?.sources || []);
    } catch (e) {
      setLocalError(e.message || String(e));
    } finally {
      setSourcesLoading(false);
    }
  };

  const installLocal = async () => {
    const path = (manualPath.trim() || picked).trim();
    if (!path) {
      setLocalError(t("installer.pathRequired"));
      return;
    }
    setInstalling(true);
    setLocalError(null);
    try {
      await api("/api/models/install-local", {
        method: "POST",
        body: JSON.stringify({ model_id: modelInfo.id, path }),
      });
      setShowLocal(false);
      await onInstalled?.();
    } catch (e) {
      setLocalError(e.message || String(e));
    } finally {
      setInstalling(false);
    }
  };

  const canDownload = Boolean(modelInfo.download_repo);
  const pct = task && task.total_bytes > 0 ? Math.min(99, Math.round((task.downloaded_bytes / task.total_bytes) * 100)) : 0;
  const showBar = task && (task.status === "downloading" || task.status === "pending");

  // A finished-but-unsuccessful task (cancelled, failed, stalled) must not remove the
  // download button. It used to render actions only when NO task had ever existed, so
  // the one interruption left the row with a status and no way to start again.
  const canStart = !showBar && !modelInfo.installed;
  const isRetry = Boolean(task) && (task.status === "error" || task.status === "cancelled");
  // The status line is ellipsis-truncated by CSS, so expose the full text on hover.
  // "Download failed" alone is undebuggable: append the backend's real reason
  // (task.error) so a failure in front of the user says WHY.
  const baseStatusText =
    task?.status === "done"
      ? t("installer.statusInstalled")
      : task?.status === "error"
        ? t("installer.statusFailed")
        : task?.status === "cancelled"
          ? t("installer.statusCancelled")
          : showBar
            ? (task?.status_text || t("installer.statusDownloading"))
            : canDownload
              ? t("installer.statusNotInstalled")
              : (t("installer.statusLocalOnly") || t("installer.statusNotInstalled"));
  const statusText =
    task?.status === "error" && task?.error
      ? `${baseStatusText}: ${task.error}`
      : baseStatusText;

  return (
    <div className={`model-installer${compact ? " compact" : ""}`}>
      <div className="model-installer-row">
        {!compact && (
          <>
            <span className={`dl-tag${task?.status ? ` ${task.status}` : ""}`}>{t("installer.tagWeights")}</span>
            <span className="model-installer-text" title={statusText}>
              {statusText}
            </span>
          </>
        )}
        {showBar ? (
          compact ? (
            <div className="model-installer-compact-bar">
              <div className="civitai-progress-track">
                <div
                  className={`civitai-progress-fill${
                    task?.status === "done" ? " done" : task?.status === "error" ? " error" : ""
                  }`}
                  style={{ transform: `scaleX(${(task?.status === "done" ? 100 : pct) / 100})` }}
                />
              </div>
              <span className="compact-bar-info">{pct}%</span>
              <button type="button" className="btn-cancel-download" onClick={cancel} title={t("app.cancel")}>
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
              </button>
            </div>
          ) : (
            <button type="button" className="btn-cancel-download" onClick={cancel} title={t("app.cancel")}>
              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
            </button>
          )
        ) : (
          <>
            {canStart && (
              <div className="model-installer-actions">
                {canDownload ? (
                  <>
                    <button
                      type="button"
                      className="btn-download-weights"
                      onClick={start}
                      disabled={starting}
                      title={t("installer.downloadBtn")}
                    >
                      {starting ? (
                        t("installer.startingBtn")
                      ) : isRetry ? (
                        t("installer.retryBtn")
                      ) : (
                        <>
                          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                            <polyline points="7 10 12 15 17 10" />
                            <line x1="12" y1="15" x2="12" y2="3" />
                          </svg>
                          <span>{t("installer.downloadBtn")}</span>
                        </>
                      )}
                    </button>
                    <button
                      type="button"
                      className={`btn-mini${showLocal ? " active" : ""}`}
                      onClick={() => (showLocal ? setShowLocal(false) : openLocal())}
                      title={t("installer.localBtnTitle")}
                    >
                      {t("installer.localBtn")}
                    </button>
                  </>
                ) : (
                  <>
                    <button
                      type="button"
                      className="btn-download-weights"
                      onClick={() => (showLocal ? setShowLocal(false) : openLocal())}
                      title={t("installer.localBtnTitle")}
                    >
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                        <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
                      </svg>
                      <span>{t("installer.selectFolderBtn")}</span>
                    </button>
                    {modelInfo?.civitai_version_id && (
                      <a
                        href={civitaiModelUrl(modelInfo.civitai_model_id, modelInfo.civitai_version_id)}
                        target="_blank"
                        rel="noreferrer"
                        className="btn-mini btn-civitai-link"
                        title={t("installer.downloadOnCivitai")}
                      >
                        Civitai ↗
                      </a>
                    )}
                  </>
                )}
              </div>
            )}
          </>
        )}
      </div>

      {!compact && showBar && (
        <div className="model-installer-progress">
          <div className="civitai-progress-track">
            <div
              className={`civitai-progress-fill${
                task?.status === "done" ? " done" : task?.status === "error" ? " error" : ""
              }`}
              // scaleX rather than width: animating width re-runs layout every tick.
              style={{ transform: `scaleX(${(task?.status === "done" ? 100 : pct) / 100})` }}
            />
          </div>
          <div className="civitai-download-footer">
            <span>{pct}%</span>
            {task?.speed_mb_s > 0 && <span className="civitai-dl-speed">{task.speed_mb_s.toFixed(1)} MB/s</span>}
            {task?.total_bytes > 0 && (
              <span>
                {(task.downloaded_bytes / 1048576).toFixed(0)} / {(task.total_bytes / 1048576).toFixed(0)} MB
              </span>
            )}
          </div>
        </div>
      )}

      {localError && !showLocal && <span className="error" role="alert">{localError}</span>}

      {showLocal && !showBar && (
        <div className="model-installer-local">
          <span className="settings-hint">{t("installer.localHint")}</span>
          {sourcesLoading ? (
            <span className="settings-hint">{t("installer.scanningCache")}</span>
          ) : (
            <select
              className="model-installer-select"
              value={picked}
              onChange={(e) => {
                setPicked(e.target.value);
                setManualPath("");
                setLocalError(null);
              }}
            >
              <option value="">
                {sources.length ? t("installer.pickCachedRepo") : t("installer.noCachedRepo")}
              </option>
              {sources.map((s) => (
                <option key={`${s.repo_id}|${s.path}`} value={s.path}>
                  {s.repo_id} — {mb(s.size_bytes)}{s.matches_model ? " (recommandé)" : ""}
                </option>
              ))}
            </select>
          )}
          <div className="path-picker-row">
            <input
              className="model-installer-path"
              type="text"
              value={manualPath}
              placeholder={t("installer.pathPlaceholder")}
              onChange={(e) => {
                setManualPath(e.target.value);
                setPicked("");
                setLocalError(null);
              }}
              spellCheck={false}
            />
            <button
              type="button"
              className="btn-mini path-picker-btn"
              onClick={handleBrowse}
              title={t("app.browse")}
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
              </svg>
              {t("app.browse")}
            </button>
          </div>
          <div className="model-installer-local-actions">
            <button
              type="button"
              className="btn-open-token"
              onClick={installLocal}
              disabled={installing || (!picked && !manualPath.trim())}
            >
              {installing ? t("installer.installingBtn") : t("installer.installLocalBtn")}
            </button>
            <button type="button" className="btn-mini" onClick={() => setShowLocal(false)}>
              {t("app.cancel")}
            </button>
          </div>
          {localError && <span className="error">{localError}</span>}
        </div>
      )}
    </div>
  );
}
