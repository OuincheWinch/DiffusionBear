import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { useI18n } from "../i18n/I18nContext";

const mb = (bytes) => `${(bytes / 1048576).toFixed(0)} MB`;

export default function ModelInstaller({ modelInfo, onInstalled }) {
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
  const pollRef = useRef(null);

  const refresh = async () => {
    try {
      const tasks = await api("/api/models/downloads");
      const matching = tasks.filter((t) => t.model_id === modelInfo.id);
      const mine = matching.find((t) => t.status === "downloading" || t.status === "pending") || matching[0] || null;
      setTask(mine);
      const done = tasks.find((t) => t.model_id === modelInfo.id && t.status === "done");
      if (done) onInstalled?.();
    } catch (e) {
      setLocalError(e.message || String(e));
    }
  };

  useEffect(() => {
    refresh();
    pollRef.current = setInterval(refresh, 1000);
    return () => clearInterval(pollRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelInfo?.id]);

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

  return (
    <div className="model-installer">
      <div className="model-installer-row">
        <span className="dl-tag">{t("installer.tagWeights")}</span>
        <span className="model-installer-text">
          {task?.status === "done"
            ? t("installer.statusInstalled")
            : task?.status === "error"
              ? t("installer.statusFailed")
              : task?.status === "cancelled"
                ? t("installer.statusCancelled")
                : showBar
                  ? `⬇ ${task?.status_text || t("installer.statusDownloading")}`
                  : t("installer.statusNotInstalled")}
        </span>
        {showBar ? (
          <button type="button" className="btn-cancel-download" onClick={cancel}>
            ✕
          </button>
        ) : (
          <>
            {!task && (
              <div className="model-installer-actions">
                {canDownload && (
                  <button type="button" className="btn-open-token" onClick={start} disabled={starting}>
                    {starting ? t("installer.startingBtn") : t("installer.downloadBtn")}
                  </button>
                )}
                <button
                  type="button"
                  className={`btn-mini${showLocal ? " active" : ""}`}
                  onClick={() => (showLocal ? setShowLocal(false) : openLocal())}
                  title={t("installer.localBtnTitle")}
                >
                  {t("installer.localBtn")}
                </button>
              </div>
            )}
            {task?.status === "error" && (
              <button type="button" className="btn-open-token" onClick={() => { setTask(null); start(); }} disabled={starting}>
                {starting ? t("installer.startingBtn") : t("app.retry")}
              </button>
            )}
          </>
        )}
      </div>

      {showBar && (
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
                  {s.matches_model ? "★ " : ""}
                  {s.repo_id} — {mb(s.size_bytes)}
                </option>
              ))}
            </select>
          )}
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
