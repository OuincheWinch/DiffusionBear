import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { useI18n } from "../i18n/I18nContext";

const POLL_MS = 1000;
const KIND_ORDER = ["diffusion", "lora", "upscaler"];
const SORTS = ["downloads", "likes", "lastModified"];

function formatCount(n) {
  if (typeof n !== "number") return "";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

function formatBytes(n) {
  if (!n) return "";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = n;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i += 1;
  }
  return `${value < 10 && i > 0 ? value.toFixed(1) : Math.round(value)} ${units[i]}`;
}

/**
 * Model Browser & Downloader.
 *
 * Searches one Hugging Face organisation and installs what it finds. Two decisions worth
 * knowing before changing anything here:
 *
 * 1. **Architecture and quantisation are filters on the repo *name*.** There is no
 *    structured field for either in the HF API, and this app's own model registry does not
 *    carry a quantisation field either - mflux infers it implicitly. So the backend does
 *    the filtering after normalising names, and returns facets computed from what it
 *    actually fetched, which is why a dropdown can never offer a value with no results.
 *
 * 2. **"Installed" is exact, and the fuzzy match is labelled separately.** The registry
 *    fetches `black-forest-labs/FLUX.2-klein-4B`, so `mlx-community/FLUX.2-Klein-4B-4bit`
 *    is a *different artefact*, not an installed model. Showing it as installed would be a
 *    lie, so the backend returns `installed` (exact) and `related_installed_as` (same
 *    model, other repo) and this renders them as different things.
 *
 * Progress reuses the existing model-download task namespace (`/api/models/downloads`),
 * so these downloads and the per-model installer on the Generate tab are the same kind of
 * object and share cancel semantics.
 */
export default function HfModelBrowser({ onInstalled }) {
  const { t } = useI18n();

  const [search, setSearch] = useState("");
  const [architecture, setArchitecture] = useState("");
  const [quantization, setQuantization] = useState("");
  const [kind, setKind] = useState("");
  const [sort, setSort] = useState("downloads");
  const [author, setAuthor] = useState("mlx-community");

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [tasks, setTasks] = useState([]);
  const [busyRepo, setBusyRepo] = useState(null);

  // Guards against a slow search response overwriting a newer one.
  const requestSeq = useRef(0);
  const mounted = useRef(true);
  useEffect(() => () => { mounted.current = false; }, []);

  const load = useCallback(async () => {
    const seq = ++requestSeq.current;
    setLoading(true);
    setError("");
    try {
      const params = new URLSearchParams();
      if (author) params.set("author", author);
      if (search.trim()) params.set("search", search.trim());
      if (architecture) params.set("architecture", architecture);
      if (quantization) params.set("quantization", quantization);
      if (kind) params.set("kind", kind);
      params.set("sort", sort);
      params.set("limit", "48");
      const data = await api(`/api/hf/models?${params.toString()}`);
      if (!mounted.current || seq !== requestSeq.current) return;
      setResult(data);
    } catch (e) {
      if (!mounted.current || seq !== requestSeq.current) return;
      setError(e.message || String(e));
    } finally {
      if (mounted.current && seq === requestSeq.current) setLoading(false);
    }
  }, [search, architecture, quantization, kind, sort, author]);

  useEffect(() => {
    load();
  }, [load]);

  // Poll only while something is actually in flight, unlike the per-model installer which
  // polls unconditionally for as long as it is mounted.
  const active = useMemo(
    () => tasks.some((task) => task.status === "downloading" || task.status === "pending"),
    [tasks],
  );

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const list = await api("/api/models/downloads");
        if (cancelled || !mounted.current) return;
        setTasks(Array.isArray(list) ? list : []);
      } catch {
        /* transient: keep the last known state */
      }
    };
    poll();
    if (!active) return undefined;
    const timer = setInterval(poll, POLL_MS);
    return () => { cancelled = true; clearInterval(timer); };
  }, [active]);

  const seenDone = useRef(new Set());
  useEffect(() => {
    for (const task of tasks) {
      if (task.status !== "done" || seenDone.current.has(task.id)) continue;
      seenDone.current.add(task.id);
      onInstalled?.();
      load();
    }
  }, [tasks, onInstalled, load]);

  const startDownload = useCallback(async (item) => {
    setBusyRepo(item.id);
    setError("");
    try {
      await api("/api/models/download", {
        method: "POST",
        body: JSON.stringify({
          model_id: item.install_name,
          repo_id: item.repo_id,
          install_name: item.install_name,
        }),
      });
      const list = await api("/api/models/downloads");
      if (mounted.current) setTasks(Array.isArray(list) ? list : []);
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      if (mounted.current) setBusyRepo(null);
    }
  }, []);

  const cancelDownload = useCallback(async (taskId) => {
    try {
      await api(`/api/models/downloads/${taskId}`, { method: "DELETE" });
      const list = await api("/api/models/downloads");
      if (mounted.current) setTasks(Array.isArray(list) ? list : []);
    } catch (e) {
      setError(e.message || String(e));
    }
  }, []);

  const facets = result?.facets || {};
  const items = result?.items || [];
  const taskByRepo = useMemo(() => {
    const map = new Map();
    for (const task of tasks) {
      if (!task.repo_id) continue;
      if (task.status === "done") continue;
      map.set(task.repo_id, task);
    }
    return map;
  }, [tasks]);

  const archOptions = facets.architecture || [];
  const quantOptions = facets.quantization || [];

  return (
    <section className="params-section hf-browser">
      <h3>{t("models.browserTitle")}</h3>
      <p className="params-section-desc">{t("models.browserDesc")}</p>

      <div className="hf-browser-controls">
        <label className="hf-browser-filter">
          <span>{t("models.filterOrg")}</span>
          <select value={author} onChange={(e) => setAuthor(e.target.value)}>
            {(result?.orgs || []).map((org) => (
              <option key={org.key} value={org.key}>{org.label}</option>
            ))}
            {/* Fallback before the first response lands, so the control is never empty. */}
            {!result && <option value="mlx-community">mlx-community</option>}
          </select>
        </label>

        <div className="hf-browser-search">
          <span className="hf-browser-search-icon" aria-hidden="true">⌕</span>
          <input
            type="search"
            value={search}
            placeholder={t("models.searchPlaceholder")}
            aria-label={t("models.searchLabel")}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>

        <label className="hf-browser-filter">
          <span>{t("models.filterArchitecture")}</span>
          <select value={architecture} onChange={(e) => setArchitecture(e.target.value)}>
            <option value="">{t("models.filterAll")}</option>
            {archOptions.map((o) => (
              <option key={o.key} value={o.key}>{o.label} ({o.count})</option>
            ))}
          </select>
        </label>

        <label className="hf-browser-filter">
          <span>{t("models.filterQuantization")}</span>
          <select value={quantization} onChange={(e) => setQuantization(e.target.value)}>
            <option value="">{t("models.filterAll")}</option>
            {quantOptions.map((o) => (
              <option key={o.key} value={o.key}>{o.label} ({o.count})</option>
            ))}
          </select>
        </label>

        <label className="hf-browser-filter">
          <span>{t("models.filterType")}</span>
          <select value={kind} onChange={(e) => setKind(e.target.value)}>
            <option value="">{t("models.filterAll")}</option>
            {KIND_ORDER.map((k) => (
              <option key={k} value={k}>{t(`models.kind.${k}`)}</option>
            ))}
          </select>
        </label>

        <label className="hf-browser-filter">
          <span>{t("models.sortBy")}</span>
          <select value={sort} onChange={(e) => setSort(e.target.value)}>
            {SORTS.map((s) => (
              <option key={s} value={s}>{t(`models.sort.${s}`)}</option>
            ))}
          </select>
        </label>
      </div>

      <div className="hf-browser-status" role="status">
        {loading
          ? t("models.searching")
          : t("models.resultsCount", { count: items.length })}
        {error && <span className="hf-browser-error">⚠ {error}</span>}
      </div>

      {!loading && items.length === 0 && !error && (
        <p className="hf-browser-empty">{t("models.noResults")}</p>
      )}

      <ul className="hf-browser-list">
        {items.map((item) => {
          const task = taskByRepo.get(item.repo_id);
          const downloading = task && task.status === "downloading";
          // Prefer the server's own progress: it is already capped at 0.99 and is the
          // value every other progress surface in the app uses. Falling back to the byte
          // ratio reproduces the old "194MB of 99MB" class of display bug.
          const pct = task
            ? Math.min(99, Math.round((task.progress || 0) * 100))
            : 0;

          return (
            <li key={item.id} className={`hf-browser-row${downloading ? " busy" : ""}`}>
              <div className="hf-browser-row-main">
                <div className="hf-browser-row-title">
                  <span className="hf-browser-name">{item.label}</span>
                  {item.installed && (
                    <span className="settings-badge ok">{t("models.installedBadge")}</span>
                  )}
                  {item.quantization && (
                    <span className="hf-chip">{item.quantization}</span>
                  )}
                  <span className="hf-chip">{item.architecture_label}</span>
                  {item.kind !== "diffusion" && (
                    <span className="hf-chip subtle">{t(`models.kind.${item.kind}`)}</span>
                  )}
                  {item.supports_alpha && (
                    <span className="hf-chip alpha">{t("models.alphaTag")}</span>
                  )}
                </div>
                <div className="hf-browser-row-meta">
                  <span className="hf-browser-repo">{item.id}</span>
                  {item.downloads > 0 && (
                    <span>{formatCount(item.downloads)} {t("models.downloads")}</span>
                  )}
                  {item.likes > 0 && (
                    <span>♥ {formatCount(item.likes)}</span>
                  )}
                  {item.related_installed_as && !item.installed && (
                    <span className="hf-browser-related">
                      {t("models.alreadyHave", { model: item.related_installed_as })}
                    </span>
                  )}
                </div>

                {downloading && (
                  <>
                    <div className="civitai-progress-track">
                      <div
                        className="civitai-progress-fill downloading"
                        style={{ transform: `scaleX(${pct / 100})` }}
                      />
                    </div>
                    <div className="civitai-download-footer">
                      <span className="civitai-dl-status-text">
                        {task.status_text || t("models.downloading")}
                      </span>
                      <span className="civitai-dl-speed">
                        {pct}% · {formatBytes(task.downloaded_bytes)}
                        {task.total_bytes > 0 && ` / ${formatBytes(task.total_bytes)}`}
                        {task.speed_mb_s > 0 && ` · ${task.speed_mb_s.toFixed(1)} MB/s`}
                      </span>
                    </div>
                  </>
                )}
              </div>

              <div className="hf-browser-row-actions">
                {downloading ? (
                  <button
                    type="button"
                    className="btn-cancel-download"
                    onClick={() => cancelDownload(task.id)}
                  >
                    {t("models.cancel")}
                  </button>
                ) : item.installed ? (
                  <span className="settings-badge ok">{t("models.installed")}</span>
                ) : (
                  <button
                    type="button"
                    className="btn-primary hf-browser-download"
                    disabled={busyRepo === item.id}
                    onClick={() => startDownload(item)}
                  >
                    {busyRepo === item.id ? t("models.preparing") : t("models.download")}
                  </button>
                )}
                <a
                  className="hf-browser-link"
                  href={`https://huggingface.co/${item.repo_id}`}
                  target="_blank"
                  rel="noreferrer noopener"
                >
                  {t("models.openOnHf")}
                </a>
              </div>
            </li>
          );
        })}
      </ul>
    </section>
  );
}