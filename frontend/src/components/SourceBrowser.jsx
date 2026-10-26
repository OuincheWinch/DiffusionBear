import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { useI18n } from "../i18n/I18nContext";
import LicenseBadges from "./LicenseBadges";

// Literal proper nouns, the same way "Hugging Face" and "Civitai" are used in App.jsx:
// these are the services' own names and translating them would misname them. "Draw
// Things" is the macOS app the folder belongs to, not a brand we may rename.
const SOURCE_LABELS = {
  huggingface: "Hugging Face",
  civitai: "Civitai",
  drawthings: "Draw Things",
  diffusionbee: "DiffusionBee",
};

const POLL_MS = 1000;
const KIND_ORDER = ["diffusion", "lora", "upscaler"];
const SORTS = ["downloads", "likes", "lastModified"];
// Civitai names its sorts differently; they cannot share one list.
// Draw Things and DiffusionBee have no sort vocabulary. The folder is a local read-only
// listing and offers nothing to order by beyond the name.
const SOURCE_SORTS = {
  drawthings: [],
  diffusionbee: [],
  huggingface: [["downloads", "models.sort.downloads"], ["likes", "models.sort.likes"], ["lastModified", "models.sort.lastModified"]],
  civitai: [
    ["Most Downloaded", "models.civitaiSort.downloads"],
    ["Highest Rated", "models.civitaiSort.rated"],
    ["Newest", "models.civitaiSort.newest"],
    ["Most Liked", "models.civitaiSort.liked"],
  ],
};

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
export default function SourceBrowser({ source: fixedSource = null, onInstalled }) {
  const { t } = useI18n();

  // Debounced mirror of the input. Civitai's cold search measures ~21s, so firing one
  // request per keystroke would queue a dozen of them; the stale-response guard discards
  // their answers but the requests still happen. Only the debounced value reaches `load`.
  const [search, setSearch] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedSearch(search), 450);
    return () => clearTimeout(timer);
  }, [search]);
  const [architecture, setArchitecture] = useState("");
  const [quantization, setQuantization] = useState("");
  const [kind, setKind] = useState("");
  const [sort, setSort] = useState("downloads");
  const [author, setAuthor] = useState("mlx-community");
  const [freeSource, setFreeSource] = useState("huggingface");
  const [sourceSort, setSourceSort] = useState("Most Downloaded");

  // A pinned source is not negotiable, so it is derived rather than mirrored into
  // state. Mirroring needs an effect to correct it, and for one render the two
  // disagree -- which is how a tab that is not on screen still gets clicked.
  const source = fixedSource || freeSource;
  const setSource = fixedSource ? () => {} : setFreeSource;

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [tasks, setTasks] = useState([]);
  const [busyRepo, setBusyRepo] = useState(null);
  const [registered, setRegistered] = useState({});
  const [detected, setDetected] = useState([]);
  // Conversion is offered only when the backend says it can actually run; otherwise the
  // button would fail 40 minutes into a 7GB download with a dependency message.
  const [convertAvailable, setConvertAvailable] = useState(null);

  // Guards against a slow search response overwriting a newer one.
  const requestSeq = useRef(0);
  const mounted = useRef(true);
  // Re-arm on every setup, not only clear on cleanup. React 19 StrictMode mounts,
  // tears down, and mounts again: with only a cleanup, the flag goes false on the
  // simulated unmount and never comes back, so every fetch result is discarded and
  // the view shows its loading state forever. Production does not double-invoke, so
  // this failed in dev only -- which is the worst possible place for it.
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);

  const load = useCallback(async () => {
    const seq = ++requestSeq.current;
    setLoading(true);
    setError("");
    try {
      if (source === "drawthings" || source === "diffusionbee") {
        // Not a search. This is the user's own folder, listed as it is. There is no
        // query to send and nothing to filter against, so the fetch is unconditional
        // and client-side filtering (below) does the narrowing.
        const endpoint = source === "drawthings"
          ? "/api/sources/drawthings/files"
          : "/api/sources/diffusionbee/files";
        const data = await api(endpoint);
        if (!mounted.current || seq !== requestSeq.current) return;
        setResult(data);
        return;
      }
      const params = new URLSearchParams();
      params.set("limit", "48");
      if (source === "civitai") {
        // Civitai filters server-side on its own vocabulary and has no
        // architecture/quantisation query params -- those are classified per item.
        if (debouncedSearch.trim()) params.set("query", debouncedSearch.trim());
        params.set("types", kind === "lora" ? "LORA" : "Checkpoint");
        params.set("sort", sourceSort);
      } else {
        if (author) params.set("author", author);
        if (debouncedSearch.trim()) params.set("search", debouncedSearch.trim());
        if (architecture) params.set("architecture", architecture);
        if (quantization) params.set("quantization", quantization);
        if (kind) params.set("kind", kind);
        params.set("sort", sort);
      }
      const data = await api(`/api/${source === "civitai" ? "civitai" : "hf"}/models?${params.toString()}`);
      if (!mounted.current || seq !== requestSeq.current) return;
      setResult(data);
    } catch (e) {
      if (!mounted.current || seq !== requestSeq.current) return;
      setResult(null);
      setError(e.message || String(e));
    } finally {
      if (mounted.current && seq === requestSeq.current) setLoading(false);
    }
  }, [debouncedSearch, architecture, quantization, kind, sort, author, source, sourceSort]);

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

  // Which engine, if any, each downloaded repo is bound to. Read from the app's own
  // model_paths override so this reflects reality rather than local optimism.
  const refreshBindings = useCallback(async () => {
    try {
      const data = await api("/api/hf/models/detected");
      if (!mounted.current) return;
      const map = {};
      for (const item of data?.items || []) map[item.name] = item.usable_as || null;
      setDetected(data?.items || []);
      const bound = await api("/api/settings");
      const paths = bound?.model_paths || {};
      const out = {};
      for (const [key, value] of Object.entries(paths)) {
        if (!value) continue;
        out[String(value).split("/").pop()] = key;
      }
      setRegistered(out);
    } catch {
      /* transient */
    }
  }, []);

  useEffect(() => {
    refreshBindings();
    api("/api/sdxl/convert/available")
      .then((d) => setConvertAvailable(d))
      .catch(() => setConvertAvailable({ available: false, reason: "" }));
  }, [refreshBindings]);

  const startConversion = useCallback(async (item) => {
    setBusyRepo(item.id);
    setError("");
    try {
      await api("/api/sdxl/convert", {
        method: "POST",
        body: JSON.stringify({ install_name: item.install_name }),
      });
      const list = await api("/api/models/downloads");
      if (mounted.current) setTasks(Array.isArray(list) ? list : []);
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      if (mounted.current) setBusyRepo(null);
    }
  }, []);

  const registerAs = useCallback(async (item, modelId) => {
    if (!item || !modelId) return;
    setBusyRepo(item.id);
    setError("");
    try {
      await api("/api/hf/models/register", {
        method: "POST",
        body: JSON.stringify({ repo_id: item.repo_id || item.name, model_id: modelId }),
      });
      await refreshBindings();
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      if (mounted.current) setBusyRepo(null);
    }
  }, [refreshBindings]);

  const unregister = useCallback(async (modelId) => {
    if (!modelId) return;
    setError("");
    try {
      await api(`/api/hf/models/register/${encodeURIComponent(modelId)}`, { method: "DELETE" });
      await refreshBindings();
    } catch (e) {
      setError(e.message || String(e));
    }
  }, [refreshBindings]);

  const startDownload = useCallback(async (item) => {
    setBusyRepo(item.id);
    setError("");
    try {
      if (item.source === "civitai") {
        const family = item.family ||
          (item.architecture === "flux3" ? "flux3" :
           item.architecture === "flux2" ? "flux2" :
           item.architecture === "pony" ? "sdxl" :
           item.architecture === "illustrious" ? "sdxl" :
           item.architecture === "krea" ? "krea" :
           item.architecture === "z-image" ? "zit" :
           item.architecture === "qwen-image" ? "qwen" : "sdxl");
        await api("/api/civitai/models/download", {
          method: "POST",
          body: JSON.stringify({
            model_id: item.civitai_model_id,
            model_version_id: item.civitai_version_id,
            name: item.label,
            family,
          }),
        });
      } else {
        await api("/api/models/download", {
          method: "POST",
          body: JSON.stringify({
            model_id: item.install_name,
            repo_id: item.repo_id,
            install_name: item.install_name,
          }),
        });
      }
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
  const allItems = useMemo(() => result?.items || [], [result]);
  const isLocalSource = source === "drawthings" || source === "diffusionbee";
  // Draw Things and DiffusionBee are local folders, not search indices, so narrowing happens here.
  // Every other source sends the query upstream and gets a pre-filtered list back.
  const items = useMemo(() => {
    if (!isLocalSource) return allItems;
    const needle = debouncedSearch.trim().toLowerCase();
    if (!needle) return allItems;
    return allItems.filter((item) => String(item.name || "").toLowerCase().includes(needle));
  }, [allItems, isLocalSource, debouncedSearch]);
  const taskByRepo = useMemo(() => {
    const map = new Map();
    for (const task of tasks) {
      if (!task.repo_id) continue;
      if (task.status === "done") continue;
      map.set(task.repo_id, task);
    }
    return map;
  }, [tasks]);

  const convertedNames = useMemo(
    () => new Set(detected.filter((d) => d.usable_as).map((d) => d.name)),
    [detected],
  );
  const archOptions = facets.architecture || [];
  const quantOptions = facets.quantization || [];

  return (
    <section className="params-section hf-browser">
      <h3>{t("models.browserTitle")}</h3>
      <p className="params-section-desc">{t("models.browserDesc")}</p>

      {!fixedSource && (
      <div className="hf-source-toggle" role="tablist">
        {/* Literal proper nouns, like the brand name in App.jsx -- "Hugging Face",
            "Civitai", "Draw Things", and "DiffusionBee" are the services' own names and must not be translated. */}
        {(fixedSource ? [[fixedSource, SOURCE_LABELS[fixedSource]]]
          : [["huggingface", SOURCE_LABELS.huggingface], ["civitai", SOURCE_LABELS.civitai],
             ["drawthings", SOURCE_LABELS.drawthings], ["diffusionbee", SOURCE_LABELS.diffusionbee]]).map(([key, label]) => (
          <button
            key={key}
            type="button"
            role="tab"
            aria-selected={source === key}
            className={`hf-source-btn${source === key ? " active" : ""}`}
            onClick={() => setSource(key)}
          >
            {label}
          </button>
        ))}
      </div>
      )}

      <div className="hf-browser-controls">
        <div className="hf-browser-search">
          <span className="hf-browser-search-icon" aria-hidden="true" style={{ display: "inline-flex", alignItems: "center" }}>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <circle cx="11" cy="11" r="8"/>
              <line x1="21" y1="21" x2="16.65" y2="16.65"/>
            </svg>
          </span>
          <input
            type="search"
            value={search}
            placeholder={source === "drawthings"
              ? t("sources.searchPlaceholder.drawthings")
              : source === "diffusionbee"
                ? t("sources.searchPlaceholder.diffusionbee")
                : source === "civitai"
                  ? t("models.searchPlaceholderCivitai")
                  : t("models.searchPlaceholder")}
            aria-label={t("models.searchLabel")}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>

        {source === "huggingface" && (
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
        )}

        {source === "huggingface" && (
          <>
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
          </>
        )}

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
          {source === "huggingface" ? (
            <select value={sort} onChange={(e) => setSort(e.target.value)}>
              {SORTS.map((s) => (
                <option key={s} value={s}>{t(`models.sort.${s}`)}</option>
              ))}
            </select>
          ) : source === "civitai" ? (
            <select value={sourceSort} onChange={(e) => setSourceSort(e.target.value)}>
              {SOURCE_SORTS.civitai.map(([value, key]) => (
                <option key={value} value={value}>{t(key)}</option>
              ))}
            </select>
          ) : null}
        </label>
      </div>

      <div className="hf-browser-status" role="status">
        {loading
          ? t("models.searching")
          : isLocalSource
            ? t("sources.detected", { count: items.length })
            : source === "civitai"
              ? t("models.resultsCountCivitai", { count: items.length })
              : t("models.resultsCount", { count: items.length })}
        {error && (
          <>
            <span className="hf-browser-error">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{ verticalAlign: "-2px", marginRight: "4px" }}><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
              {error}
            </span>
            {/* Civitai in particular is slow and flaky; a retry button beats a dead spinner. */}
            <button type="button" className="btn-mini" onClick={load}>
              {t("models.retry")}
            </button>
          </>
        )}
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
            <li
              // `id` exists on HF and Civitai items but NOT on a Draw Things FileInfo
              // (name/size_bytes/modified_at/base_model/kind only), so keying on it gave
              // every local row `undefined` -- duplicate keys, and React reusing DOM
              // across rows whenever the client-side filter reorders the list.
              key={item.repo_id || item.name}
              className={`hf-browser-row${downloading ? " busy" : ""}`}
            >
              <div className="hf-browser-row-main">
                <div className="hf-browser-row-title">
                  <span className="hf-browser-name">{item.label || item.name}</span>
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
                  {isLocalSource && item.kind === "model" && (
                    <span className="hf-chip subtle" title={t("sources.modelsComingSoon")}>
                      {t("sources.modelsComingSoon")}
                    </span>
                  )}
                  {!item.usable_as && !convertedNames.has(item.install_name) && (
                    <span className="hf-chip notrunnable" title={item.usable_as_label || ""}>
                      {t("models.notRunnable")}
                    </span>
                  )}
                  {convertedNames.has(item.install_name) && (
                    <span className="settings-badge ok">{t("models.convertedBadge")}</span>
                  )}
                </div>
                {/* Rendered from whatever the cached payload already carries. A list
                    render never fetches: an item with no licence simply shows no badge,
                    and the row appears immediately rather than waiting on a request. */}
                {item.license && (
                  <LicenseBadges license={item.license} compact />
                )}
                <div className="hf-browser-row-meta">
                  <span className="hf-browser-repo">{item.repo_id}</span>
                  {item.creator && <span>{item.creator}</span>}
                  {item.size_bytes > 0 && <span>{formatBytes(item.size_bytes)}</span>}
                  {item.downloads > 0 && (
                    <span>{formatCount(item.downloads)} {t("models.downloads")}</span>
                  )}
                  {item.likes > 0 && (
                    <span style={{ display: "inline-flex", alignItems: "center" }}>
                      <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{ marginRight: "3px" }}><path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/></svg>
                      {formatCount(item.likes)}
                    </span>
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
                  <>
                    <span className="settings-badge ok">{t("models.installed")}</span>
                    {!convertedNames.has(item.install_name) && item.source === "civitai" && (
                      <button
                        type="button"
                        className="btn-mini"
                        disabled={busyRepo === item.id || convertAvailable?.available === false}
                        title={convertAvailable?.reason || ""}
                        onClick={() => startConversion(item)}
                      >
                        {convertAvailable?.available === false
                          ? t("models.convertUnavailable")
                          : t("models.convert")}
                      </button>
                    )}
                    {item.usable_as && registered[item.install_name] !== item.usable_as ? (
                      <button
                        type="button"
                        className="btn-mini"
                        disabled={busyRepo === item.id}
                        onClick={() => registerAs(item, item.usable_as)}
                      >
                        {t("models.useFor", { model: item.usable_as_label })}
                      </button>
                    ) : item.usable_as && registered[item.install_name] === item.usable_as ? (
                      <button
                        type="button"
                        className="btn-mini"
                        onClick={() => unregister(item.usable_as)}
                      >
                        {t("models.stopUsing", { model: item.usable_as_label })}
                      </button>
                    ) : (
                      <span className="hf-chip notrunnable" title={item.usable_as_label || ""}>
                        {t("models.notRunnable")}
                      </span>
                    )}
                  </>
                ) : isLocalSource ? (
                  // A local folder has no repo to download and no model id to register
                  // it under; `startDownload` would post an empty body and get a 422.
                  // Full models are out of scope; LoRAs are imported from the Import
                  // side of this tab, which is where that flow belongs.
                  <span className="hf-chip subtle">
                    {item.kind === "model"
                      ? t("sources.modelsComingSoon")
                      : t("sources.importOnImportTab")}
                  </span>
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
                  href={
                    source === "civitai"
                      ? `https://civitai.com/models/${item.civitai_model_id}`
                      : `https://huggingface.co/${item.repo_id}`
                  }
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