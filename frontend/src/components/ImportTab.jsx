import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";
import { useI18n } from "../i18n/I18nContext";
import LicenseBadges from "./LicenseBadges";

/**
 * Import a LoRA out of the Draw Things or DiffusionBee folders.
 *
 * Three states, in order, and the user is always told which one they are in:
 *
 *   list    what is in the folder, read-only, with the detected base model
 *   convert the file is being rewritten; then the review form appears
 *   review  name, scale and trigger, plus a Civitai match the user chose themselves
 *
 * **The skipped-tensor count is shown, not swallowed.** A FLUX LoRA carries norm2
 * modulation with no destination module: 114 tensors on the measured file. Silently
 * dropping them would produce a LoRA that quietly does less than the one in the
 * folder, and the user would find out from the output rather than from this screen.
 *
 * **Nothing is written to the source folders.** They are the user's own libraries and
 * this app has no business mutating them; endpoints enforce that server-side, and
 * this component never offers a path that would try.
 *
 * A Civitai match is never applied automatically. The endpoint returns candidates
 * with `attached: false`, and only a click here attaches one. Picking the wrong
 * licence silently is worse than having none.
 */

function formatBytes(n) {
  if (!n) return "";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = n;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i += 1;
  }
  return `${value.toFixed(value < 10 && i > 0 ? 1 : 0)} ${units[i]}`;
}

const IMPORT_SOURCES = [
  ["drawthings", "Draw Things"],
  ["diffusionbee", "DiffusionBee"],
];

// The registry's vocabulary, not the converter's.
//
// `/api/sources/drawthings/convert` speaks `flux` -- its request validates
// `^(sdxl|flux)$` and the summary reports that value back. `POST /api/loras` speaks
// `flux2`; `_BASE_MODELS` is {sdxl, flux2, krea2, z-image} and anything else is a 400.
// Passing the converter's value straight through made EVERY FLUX import fail at Confirm,
// after the ~40s conversion had already run. `BaseModelVocabularyTests` in
// backend/test_review_bugs.py pins both vocabularies so this cannot drift apart quietly.
const REGISTRY_BASE_MODEL = { sdxl: "sdxl", flux: "flux2" };

// "add_detail_xl_lora_f16.ckpt" -> "add detail xl lora f16". Prefilling the review
// form beats an empty box, and the user still owns the final value.
function prettyName(filename) {
  return String(filename || "")
    .replace(/\.(ckpt|tdict)$/i, "")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

// The registry validates `^[A-Za-z0-9][A-Za-z0-9 ._-]{0,199}$`. Civitai and Japanese
// model names carry parentheses, commas, plus signs and non-ASCII, all of which are
// rejected -- so a name that looked fine in the review box failed at save, after the
// conversion had run. Fold anything outside the accepted set into spaces, then collapse.
const REGISTRY_NAME_MAX = 200;

function safeName(text) {
  const folded = String(text || "")
    // eslint-disable-next-line no-control-regex
    .replace(/[^A-Za-z0-9 ._-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, REGISTRY_NAME_MAX);
  // Must not start with the separator class; fall back to a stable transliteration.
  return /^[A-Za-z0-9]/.test(folded) ? folded : folded.replace(/^[ ._-]+/, "") || "LoRA";
}

export default function ImportTab({ onImported } = {}) {
  const { t } = useI18n();

  const [source, setSource] = useState("drawthings");
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [converting, setConverting] = useState(null);
  const [result, setResult] = useState(null);
  const [name, setName] = useState("");
  const [scale, setScale] = useState(1);
  const [trigger, setTrigger] = useState("");
  const [saving, setSaving] = useState(false);

  const [matchQuery, setMatchQuery] = useState("");
  const [matching, setMatching] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [candidates, setCandidates] = useState([]);
  const [chosen, setChosen] = useState(null);

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
    setLoading(true);
    setError("");
    try {
      const data = source === "diffusionbee"
        ? await api("/api/sources/diffusionbee/files")
        : await api("/api/sources/drawthings/files");
      if (mounted.current) setFiles(data?.items || []);
    } catch (e) {
      if (mounted.current) setError(e.message || String(e));
    } finally {
      if (mounted.current) setLoading(false);
    }
  }, [source]);

  useEffect(() => { load(); }, [load]);

  const startConvert = useCallback(async (file, baseModel) => {
    setConverting(file.name);
    setError("");
    setResult(null);
    setCandidates([]);
    setChosen(null);
    setMatchQuery("");
    try {
      const data = source === "diffusionbee"
        ? await api("/api/sources/diffusionbee/convert", {
            method: "POST",
            body: JSON.stringify({ filename: file.name, base_model: baseModel }),
          })
        : await api("/api/sources/drawthings/convert", {
            method: "POST",
            body: JSON.stringify({ filename: file.name, base_model: baseModel }),
          });
      if (!mounted.current) return;
      setResult(data);
      setName(prettyName(file.name));
      setScale(1);
      setTrigger("");
    } catch (e) {
      if (mounted.current) setError(e.message || String(e));
    } finally {
      if (mounted.current) setConverting(null);
    }
  }, [source]);

  const match = useCallback(async () => {
    setMatching(true);
    setError("");
    try {
      const payload = matchQuery.trim().startsWith("http")
        ? { version_url: matchQuery.trim() }
        : { query: matchQuery.trim() };
      const data = source === "diffusionbee"
        ? await api("/api/sources/diffusionbee/civitai-match", {
            method: "POST",
            body: JSON.stringify(payload),
          })
        : await api("/api/sources/drawthings/civitai-match", {
            method: "POST",
            body: JSON.stringify(payload),
          });
      if (!mounted.current) return;
      setCandidates(data?.candidates || []);
      setHasSearched(true);
    } catch (e) {
      if (mounted.current) setError(e.message || String(e));
    } finally {
      if (mounted.current) setMatching(false);
    }
  }, [source, matchQuery]);

  const confirm = useCallback(async () => {
    if (!result) return;
    setSaving(true);
    setError("");
    try {
      const body = {
        name: safeName(name),
        path: result.output_path,
        scale,
        base_model: REGISTRY_BASE_MODEL[result.base_model] || result.base_model,
      };
      if (trigger.trim()) body.triggers = [trigger.trim()];
      // Only attached when the user picked a candidate. Absent otherwise, so the
      // registry omits the key rather than storing an empty licence block.
      if (chosen?.license) body.license = chosen.license;

      await api("/api/loras", { method: "POST", body: JSON.stringify(body) });

      // Announce it. Both writes have landed by here -- file and registry entry -- but
      // nothing else in the app has asked to see them. `load()` below re-reads the
      // DRAW THINGS folder listing, which this import did not change, so it never
      // refreshed the registry the user is about to look for their LoRA in.
      //
      // The registry lives in useLoraPanel inside GenerateForm, a different component in
      // a different tab, and this app keeps every tab mounted, so it was fetched once on
      // mount and never again. Refreshing it here is what stops "import, see nothing,
      // reboot, it's there". The window event follows the existing mlx:* convention
      // (mlx:load-prompt, mlx:cancel-job); the prop is for the caller that already owns
      // a registry refresh of its own.
      window.dispatchEvent(new CustomEvent("mlx:loras-changed"));
      onImported?.();

      if (!mounted.current) return;
      setResult(null);
      setChosen(null);
      await load();
    } catch (e) {
      if (mounted.current) setError(e.message || String(e));
    } finally {
      if (mounted.current) setSaving(false);
    }
  }, [result, name, scale, trigger, chosen, load, onImported]);

  const cancel = useCallback(() => {
    setResult(null);
    setCandidates([]);
    setChosen(null);
    setHasSearched(false);
    setError("");
  }, []);

  return (
    <section className="params-section import-tab">
      <h3>{t("sources.title")}</h3>

      {error && (
        <div className="settings-feedback error" role="alert">
          ⚠ {error}
        </div>
      )}

      {!result && (
        <>
          <div className="hf-source-toggle" role="tablist">
            {IMPORT_SOURCES.map(([key, label]) => (
              <button
                key={key}
                type="button"
                role="tab"
                aria-selected={source === key}
                className={`hf-source-btn${source === key ? " active" : ""}`}
                onClick={() => {
                  if (source !== key) {
                    setSource(key);
                    setError("");
                  }
                }}
              >
                {label}
              </button>
            ))}
          </div>

          <ul className="import-list">
          {loading && <li className="import-empty">{t("models.searching")}</li>}
          {!loading && files.length === 0 && (
            <li className="import-empty">{t("models.noResults")}</li>
          )}
          {files.map((file) => {
            const isModel = file.kind === "model";
            const base = file.base_model;
            return (
              <li key={file.name} className="import-row">
                <div className="import-row-main">
                  <span className="import-name">{file.name}</span>
                  <span className="import-meta">
                    {formatBytes(file.size_bytes)}
                    {file.modified_at
                      ? ` · ${new Date(file.modified_at * 1000).toLocaleDateString()}`
                      : ""}
                  </span>
                  {base ? (
                    <span className="hf-chip">{t("sources.baseModel")}: {base}</span>
                  ) : (
                    <span className="hf-chip subtle">{t("sources.unknownLicense")}</span>
                  )}
                  {isModel && (
                    <span className="hf-chip subtle">{t("sources.modelsComingSoon")}</span>
                  )}
                  {base === "flux" && (
                    <span className="import-warning">{t("sources.fluxNoModelWarning")}</span>
                  )}
                </div>
                <div className="import-row-actions">
                  {isModel || !base ? (
                    <span className="import-disabled">{t("sources.modelsComingSoon")}</span>
                  ) : (
                    <button
                      type="button"
                      className="btn-mini"
                      disabled={converting === file.name}
                      onClick={() => startConvert(file, base)}
                    >
                      {converting === file.name ? t("sources.importing") : t("sources.import")}
                    </button>
                  )}
                </div>
              </li>
            );
          })}
        </ul>
        </>
      )}

      {result && (
        <div className="import-review">
          <h4>{t("sources.review")}</h4>

          {/* Review Focus 3. Rendered whenever the count is above zero, from the
              response -- never recomputed, never rounded away, never omitted because
              the import still succeeded. */}
          {result.skipped_tensors > 0 && (
            <div className="import-warning" role="status">
              {t("sources.skippedTensors", { count: result.skipped_tensors })}
            </div>
          )}

          <label className="import-field">
            <span>{t("sources.name")}</span>
            <input type="text" value={name} onChange={(e) => setName(e.target.value)} />
          </label>

          <label className="import-field">
            <span>{t("sources.scale")}</span>
            <input
              type="number" min="0" max="10" step="0.05"
              value={scale}
              onChange={(e) => setScale(Number(e.target.value))}
            />
          </label>

          <label className="import-field">
            <span>{t("sources.trigger")}</span>
            <input
              type="text" value={trigger}
              onChange={(e) => setTrigger(e.target.value)}
            />
          </label>

          <div className="import-match">
            <h4>{t("sources.matchCivitai")}</h4>
            <div className="import-match-row">
              <input
                type="text"
                value={matchQuery}
                placeholder={t("sources.matchByUrl")}
                onChange={(e) => setMatchQuery(e.target.value)}
              />
              <button
                type="button" className="btn-mini"
                disabled={matching || !matchQuery.trim()}
                onClick={match}
              >
                {t("sources.matchCivitai")}
              </button>
            </div>

            {/* Only after a search has actually run: while the user is still typing,
                matchQuery is truthy and this said "no match" for every keystroke. */}
            {hasSearched && candidates.length === 0 && !matching && (
              <p className="import-empty">{t("sources.noMatch")}</p>
            )}

            {candidates.map((cand) => {
              // A Civitai model has several versions, so identity is the pair. There is
              // no `id` and no `name` on a candidate: the version name is what the
              // creator called it, and it is frequently blank, hence the fallback.
              const key = `${cand.model_id}:${cand.version_id}`;
              const active = chosen && `${chosen.model_id}:${chosen.version_id}` === key;
              return (
                <button
                  type="button"
                  key={key}
                  className={`import-candidate${active ? " active" : ""}`}
                  onClick={() => setChosen(active ? null : cand)}
                >
                  <span className="import-candidate-name">
                    {cand.version_name || cand.model_id}
                  </span>
                  <LicenseBadges license={cand.license} compact />
                </button>
              );
            })}

            {chosen && <LicenseBadges license={chosen.license} compact />}
          </div>

          <div className="import-actions">
            <button
              type="button" className="btn-mini primary"
              disabled={saving || !name.trim()}
              onClick={confirm}
            >
              {saving ? t("sources.importing") : t("sources.confirm")}
            </button>
            <button type="button" className="btn-mini" onClick={cancel}>
              {t("sources.cancel")}
            </button>
          </div>
        </div>
      )}
    </section>
  );
}