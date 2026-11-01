import { useCallback, useEffect, useRef, useState } from "react";
import { api, pickPath } from "../api";
import { useI18n } from "../i18n/I18nContext";
import { formatBytes } from "../utils/formatBytes";

/**
 * Onboard a model the registry never listed: a local folder, any Hugging Face
 * repo id, or a Civitai model version -- provided it belongs to a supported
 * family (SDXL, FLUX.2, Krea, Z-Image Turbo, Qwen-Image).
 *
 * A separate component from ImportTab on purpose: ImportTab is Draw
 * Things-specific and its review flow is conversion-shaped; mixing the two
 * flows risks regressions in a working tab.
 *
 * Declare -> review -> confirm, never auto-attach (spec section 2):
 *
 *   declare  family picker (+variant for FLUX.2 / Qwen) plus one source field
 *   review   the declaration laid out for a last look
 *   confirm  local registers at once; HF/Civitai start the existing download
 *            workers with the family declaration and register when the task
 *            finishes. A Civitai version's baseModel tag may PREFILL the family
 *            where present -- the declaration still wins, never the tag.
 *
 * Success fires `mlx:models-changed` (mirroring the `mlx:loras-changed`
 * convention) so the Generate picker refreshes with no restart.
 */

const FAMILIES = ["sdxl", "flux2", "krea", "zit", "qwen"];
const VARIANTS = { flux2: ["4b", "9b"], qwen: ["base", "turbo"] };
const SOURCES = ["local", "hf", "civitai"];

// Consecutive status-poll failures tolerated before the download is declared
// unwedgeable. The poll is the only thing that can notice a stalled task, so a
// failure here must surface, never swallow: a silent catch wedges the spinner
// with no error and no timeout.
const MAX_POLL_FAILURES = 3;

// Model pages, mirror domains, and download links -- anything carrying the ids:
// "https://civitai.com/models/12345/my-model?modelVersionId=67890&ref_code=88C8VEBA",
// "https://civitai.com/api/download/models/67890", or a bare version id.
// Only numbers are ever extracted; the URL itself is never fetched (the
// backend resolves everything by id against the official API), so a mirror
// or lookalike domain cannot smuggle a destination in. Returns null when
// neither id is present. A missing modelId is fine -- the confirm step
// resolves it through prefill -- but a missing versionId has nothing to act on.
function parseCivitaiRef(text) {
  const raw = String(text || "").trim();
  if (/^\d+$/.test(raw)) return { modelId: null, versionId: Number(raw) };
  const model = raw.match(/civitai\.(?:com|red)\/models\/(\d+)/i);
  const version = raw.match(/[?&]modelVersionId=(\d+)/i)
    || raw.match(/api\/download\/models\/(\d+)/i);
  if (!version) return null;
  return {
    modelId: model ? Number(model[1]) : null,
    versionId: Number(version[1]),
  };
}

export default function OnboardModelTab({ onRegistered } = {}) {
  const { t } = useI18n();

  const [label, setLabel] = useState("");
  const [family, setFamily] = useState("sdxl");
  const [variant, setVariant] = useState("");
  const [source, setSource] = useState("local");
  const [localPath, setLocalPath] = useState("");
  const [hfRepo, setHfRepo] = useState("");
  const [civitaiRef, setCivitaiRef] = useState("");
  const [suggested, setSuggested] = useState(false);
  const [reviewing, setReviewing] = useState(false);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState("");
  const [done, setDone] = useState("");
  const [taskId, setTaskId] = useState(null);
  const [activeTask, setActiveTask] = useState(null);
  const [singleFileError, setSingleFileError] = useState(false);

  const mounted = useRef(true);
  // Re-arm on every setup, not only clear on cleanup: React 19 StrictMode's
  // simulated unmount would otherwise leave the flag false and discard every
  // fetch result in dev. Same shape as ImportTab.
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);

  const variants = VARIANTS[family] || [];

  function pickFamily(next) {
    setFamily(next);
    setVariant(VARIANTS[next] ? VARIANTS[next][0] : "");
    setSuggested(false);
  }

  const suggest = useCallback(async () => {
    const parsed = parseCivitaiRef(civitaiRef);
    if (!parsed) return;
    setWorking(true);
    setError("");
    try {
      const data = await api(`/api/civitai/models/version/${parsed.versionId}/prefill`);
      if (!mounted.current) return;
      if (data?.suggested_family && FAMILIES.includes(data.suggested_family)) {
        setFamily(data.suggested_family);
        setVariant(VARIANTS[data.suggested_family] ? VARIANTS[data.suggested_family][0] : "");
        setSuggested(true);
      }
    } catch (e) {
      if (mounted.current) setError(e.message || String(e));
    } finally {
      if (mounted.current) setWorking(false);
    }
  }, [civitaiRef]);

  // A started HF/Civitai download registers itself server-side when the worker
  // tail finishes; the client watches its own task and announces the arrival.
  // The poll runs only while a task this component started is in flight.
  const pollFailures = useRef(0);
  useEffect(() => {
    if (!taskId) return undefined;
    pollFailures.current = 0;
    const timer = setInterval(async () => {
      try {
        const list = await api("/api/models/downloads");
        if (!mounted.current) return;
        pollFailures.current = 0;
        const task = (Array.isArray(list) ? list : []).find((item) => item.id === taskId);
        if (!task) return;
        setActiveTask(task);
        if (task.status === "done") {
          clearInterval(timer);
          setTaskId(null);
          setActiveTask(null);
          setWorking(false);
          window.dispatchEvent(new CustomEvent("mlx:models-changed"));
          onRegistered?.();
          setDone(t("models.onboard.registered", { model: task.model_name || label || taskId }));
          setReviewing(false);
        } else if (task.status === "error" || task.status === "cancelled") {
          clearInterval(timer);
          setTaskId(null);
          setActiveTask(null);
          setWorking(false);
          setError(task.error || task.status);
        }
      } catch (e) {
        // Transient, until it is not: after MAX_POLL_FAILURES consecutive
        // failures the task status is unknowable, so stop polling and say so
        // in the error box rather than spinning forever.
        pollFailures.current += 1;
        if (mounted.current && pollFailures.current >= MAX_POLL_FAILURES) {
          clearInterval(timer);
          setTaskId(null);
          setActiveTask(null);
          setWorking(false);
          setError(e.message || String(e));
        }
      }
    }, 2000);
    return () => clearInterval(timer);
  }, [taskId, label, onRegistered, t]);

  const cancelDownload = useCallback(async () => {
    if (!taskId) return;
    try {
      await api(`/api/models/downloads/${taskId}`, { method: "DELETE" });
      setTaskId(null);
      setActiveTask(null);
      setWorking(false);
    } catch (e) {
      setError(e.message || String(e));
    }
  }, [taskId]);

  const convertSingleFile = useCallback(async () => {
    setWorking(true);
    setError("");
    setDone("");
    setSingleFileError(false);
    try {
      const name = label.trim() || undefined;
      const res = await api("/api/models/onboard", {
        method: "POST",
        body: JSON.stringify({
          family: "sdxl",
          local_path: localPath.trim(),
          ...(name ? { label: name } : {}),
          convert_if_single_file: true,
        }),
      });
      if (!mounted.current) return;
      if (res?.task_id) {
        setTaskId(res.task_id);
        setDone(t("models.onboard.convertingStarted"));
      } else if (res?.status === "already_converted" || res?.id) {
        window.dispatchEvent(new CustomEvent("mlx:models-changed"));
        onRegistered?.();
        setDone(t("models.onboard.registered", { model: res?.label || res?.id || name || localPath.trim() }));
        setReviewing(false);
        setWorking(false);
      }
    } catch (e) {
      if (mounted.current) setError(e.message || String(e));
      if (mounted.current) setWorking(false);
    }
  }, [label, localPath, onRegistered, t]);

  const confirm = useCallback(async () => {
    setWorking(true);
    setError("");
    setDone("");
    setSingleFileError(false);
    try {
      const name = label.trim() || undefined;
      const declaredVariant = VARIANTS[family] ? (variant || VARIANTS[family][0]) : undefined;
      if (source === "local") {
        const body = {
          family,
          local_path: localPath.trim(),
          ...(family === "sdxl" ? { convert_if_single_file: true } : {}),
        };
        if (name) body.label = name;
        if (declaredVariant) body.variant = declaredVariant;
        const entry = await api("/api/models/onboard", {
          method: "POST",
          body: JSON.stringify(body),
        });
        if (!mounted.current) return;
        if (entry?.status === "converting" && entry?.task_id) {
          setTaskId(entry.task_id);
          setDone(t("models.onboard.convertingStarted"));
          return;
        }
        // Announce it. The entry landed with this POST, but the Generate picker
        // fetched /api/models once on mount and would otherwise stay stale
        // until restart -- the same gap the mlx:loras-changed event closes for
        // LoRAs. The prop refreshes the Models tab's own table.
        window.dispatchEvent(new CustomEvent("mlx:models-changed"));
        onRegistered?.();
        setDone(t("models.onboard.registered", { model: entry?.label || entry?.id || name || localPath.trim() }));
        setReviewing(false);
        setWorking(false);
      } else if (source === "hf") {
        const repo = hfRepo.trim();
        const started = await api("/api/models/download", {
          method: "POST",
          body: JSON.stringify({
            model_id: name || repo,
            repo_id: repo,
            ...(name ? { install_name: name } : {}),
            family,
            ...(declaredVariant ? { variant: declaredVariant } : {}),
          }),
        });
        if (!mounted.current) return;
        setTaskId(started?.task_id || null);
        setDone(t("models.onboard.downloadStarted"));
        if (!started?.task_id) setWorking(false);
      } else {
        const parsed = parseCivitaiRef(civitaiRef);
        if (!parsed?.versionId) {
          throw new Error(t("models.onboard.civitaiRef"));
        }
        // Version-only inputs (bare ids, download links) carry no model id.
        // Resolve it through prefill -- the same endpoint that suggested the
        // family -- before starting the download that requires both.
        let modelId = parsed.modelId;
        if (modelId == null) {
          const prefill = await api(`/api/civitai/models/version/${parsed.versionId}/prefill`);
          if (!mounted.current) return;
          modelId = prefill?.model_id ?? null;
          if (modelId == null) {
            throw new Error(t("models.onboard.civitaiRef"));
          }
        }
        const started = await api("/api/civitai/models/download", {
          method: "POST",
          body: JSON.stringify({
            model_id: modelId,
            model_version_id: parsed.versionId,
            ...(name ? { name } : {}),
            family,
            ...(declaredVariant ? { variant: declaredVariant } : {}),
          }),
        });
        if (!mounted.current) return;
        setTaskId(started?.task_id || null);
        setDone(t("models.onboard.downloadStarted"));
        if (!started?.task_id) setWorking(false);
      }
    } catch (e) {
      const msg = e.message || String(e);
      if (mounted.current) {
        setError(msg);
        if (family === "sdxl" && msg.toLowerCase().includes("single-file")) {
          setSingleFileError(true);
        }
        setWorking(false);
      }
    }
  }, [label, family, variant, source, localPath, hfRepo, civitaiRef, onRegistered, t]);

  const handleBrowseLocal = async () => {
    try {
      const res = await pickPath({ kind: "folder", title: t("models.onboard.localPath") });
      if (!res.cancelled && res.path) {
        setLocalPath(res.path);
        if (!label) {
          const parts = res.path.split("/").filter(Boolean);
          if (parts.length > 0) setLabel(parts[parts.length - 1]);
        }
      }
    } catch (e) {
      setError(e.message || String(e));
    }
  };

  const canReview =
    (source === "local" && localPath.trim()) ||
    (source === "hf" && hfRepo.trim()) ||
    (source === "civitai" && civitaiRef.trim());

  const sourceKey = { local: "models.onboard.sourceLocal", hf: "models.onboard.sourceHf", civitai: "models.onboard.sourceCivitai" }[source];

  return (
    <section className="params-section onboard-tab">
      <h3>{t("models.onboard.title")}</h3>
      <p className="params-section-desc">{t("models.onboard.desc")}</p>

      {error && (
        <div className="settings-feedback error" role="alert">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{ verticalAlign: "-2px", marginRight: "6px" }}><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
          {error}
        </div>
      )}
      {singleFileError && (
        <div className="import-actions" style={{ marginBottom: "1rem" }}>
          <button
            type="button"
            className="btn-mini primary"
            disabled={working}
            onClick={convertSingleFile}
          >
            {working ? t("models.onboard.working") : t("models.onboard.convertAndRegister")}
          </button>
        </div>
      )}
      {done && (
        <div className="settings-feedback success" role="status">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{ verticalAlign: "-2px", marginRight: "6px" }}><polyline points="20 6 9 17 4 12"/></svg>
          {done}
        </div>
      )}

      {activeTask && (activeTask.status === "downloading" || activeTask.status === "pending") && (
        <div className="model-installer-progress" style={{ margin: "1rem 0" }}>
          <div className="civitai-progress-track">
            <div
              className="civitai-progress-fill"
              style={{
                transform: `scaleX(${
                  activeTask.progress != null
                    ? Math.min(0.99, Math.max(0.01, activeTask.progress))
                    : 0
                })`,
              }}
            />
          </div>
          <div className="civitai-download-footer" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "0.5rem" }}>
            <span className="civitai-dl-status-text">
              {activeTask.status_text || t("models.onboard.working")}
            </span>
            <span className="civitai-dl-speed">
              {Math.min(99, Math.round((activeTask.progress || 0) * 100))}%
              {activeTask.downloaded_bytes > 0 && ` · ${formatBytes(activeTask.downloaded_bytes)}`}
              {activeTask.total_bytes > 0 && ` / ${formatBytes(activeTask.total_bytes)}`}
              {activeTask.speed_mb_s > 0 && ` · ${activeTask.speed_mb_s.toFixed(1)} MB/s`}
            </span>
            <button
              type="button"
              className="btn-mini"
              onClick={cancelDownload}
            >
              {t("sources.cancel")}
            </button>
          </div>
        </div>
      )}

      {!reviewing && (
        <>
          <label className="import-field">
            <span>{t("models.onboard.label")}</span>
            <input type="text" value={label} onChange={(e) => setLabel(e.target.value)} />
          </label>

          <label className="import-field">
            <span>{t("models.onboard.family")}</span>
            <select value={family} onChange={(e) => pickFamily(e.target.value)}>
              {FAMILIES.map((item) => (
                <option key={item} value={item}>{item}</option>
              ))}
            </select>
          </label>
          {suggested && (
            <p className="import-empty" role="status">{t("models.onboard.suggested")}</p>
          )}

          {variants.length > 0 && (
            <label className="import-field">
              <span>{t("models.onboard.variant")}</span>
              <select value={variant || variants[0]} onChange={(e) => setVariant(e.target.value)}>
                {variants.map((item) => (
                  <option key={item} value={item}>{item}</option>
                ))}
              </select>
            </label>
          )}

          <label className="import-field">
            <span>{t("models.onboard.source")}</span>
            <select value={source} onChange={(e) => { setSource(e.target.value); setReviewing(false); }}>
              {SOURCES.map((item) => (
                <option key={item} value={item}>
                  {t({ local: "models.onboard.sourceLocal", hf: "models.onboard.sourceHf", civitai: "models.onboard.sourceCivitai" }[item])}
                </option>
              ))}
            </select>
          </label>

          {source === "local" && (
            <label className="import-field">
              <span>{t("models.onboard.localPath")}</span>
              <div className="path-picker-row">
                <input type="text" value={localPath} onChange={(e) => setLocalPath(e.target.value)} />
                <button
                  type="button"
                  className="btn-mini path-picker-btn"
                  onClick={handleBrowseLocal}
                  title={t("app.browse")}
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
                  </svg>
                  {t("app.browse")}
                </button>
              </div>
            </label>
          )}
          {source === "hf" && (
            <label className="import-field">
              <span>{t("models.onboard.hfRepo")}</span>
              <input type="text" value={hfRepo} onChange={(e) => setHfRepo(e.target.value)} />
            </label>
          )}
          {source === "civitai" && (
            <>
              <label className="import-field">
                <span>{t("models.onboard.civitaiRef")}</span>
                <input type="text" value={civitaiRef} onChange={(e) => { setCivitaiRef(e.target.value); setSuggested(false); }} />
              </label>
              <button type="button" className="btn-mini" disabled={working || !parseCivitaiRef(civitaiRef)} onClick={suggest}>
                {working ? t("models.onboard.working") : t("models.onboard.suggest")}
              </button>
            </>
          )}

          <div className="import-actions">
            <button type="button" className="btn-mini primary" disabled={!canReview} onClick={() => setReviewing(true)}>
              {t("models.onboard.review")}
            </button>
          </div>
        </>
      )}

      {reviewing && (
        <div className="import-review">
          <h4>{t("models.onboard.review")}</h4>
          <p className="import-empty">
            {family}
            {variants.length > 0 ? ` · ${variant || variants[0]}` : ""}
            {` · ${t(sourceKey)}`}
          </p>
          <p className="import-empty">
            {source === "local" ? localPath.trim() : source === "hf" ? hfRepo.trim() : civitaiRef.trim()}
          </p>
          <div className="import-actions">
            {singleFileError ? (
              <button type="button" className="btn-mini primary" disabled={working} onClick={convertSingleFile}>
                {working ? t("models.onboard.working") : t("models.onboard.convertAndRegister")}
              </button>
            ) : (
              <button type="button" className="btn-mini primary" disabled={working} onClick={confirm}>
                {working ? t("models.onboard.working") : t("models.onboard.confirm")}
              </button>
            )}
            <button type="button" className="btn-mini" onClick={() => setReviewing(false)}>
              {t("sources.cancel")}
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
