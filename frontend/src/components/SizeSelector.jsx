import { useEffect, useMemo, useRef, useState } from "react";
import { useI18n } from "../i18n/I18nContext";
import {
  ORIENTATIONS,
  MULTIPLE,
  NATIVE_STEP,
  snap,
  fitToBudget,
  presetsFor,
  allAllowed,
  orientationFor,
  orientationOf,
  ratioLabel,
  overAdvisory,
  advisoryFor,
} from "./sizePresets";

// SizeSelector -- orientation tab, then one native preset, or Custom.
//
// The previous control chose a base resolution and an aspect ratio separately
// and multiplied them, which produced sizes no model was trained on (16:9 at
// base 1024 is 1024x576, and it looks it). The ratio is now a property of the
// resolution: this control only ever emits a pair a model recognises, plus
// whatever the user types in Custom.
//
// Props are unchanged from the old control -- GenerateForm passes width,
// setWidth, height, setHeight, maxPixels, modelInfo -- so nothing at the call
// site had to move.
export default function SizeSelector({
  width,
  setWidth,
  height,
  setHeight,
  maxPixels,
  modelInfo,
}) {
  const { t } = useI18n();


  const allowed = useMemo(() => allAllowed(modelInfo), [modelInfo]);
  const allowedKey = useMemo(
    () => allowed.map((p) => `${p.width}x${p.height}`).join(","),
    [allowed],
  );

  // Adopt the incoming size whenever it comes from outside (hydration, model
  // switch, reuse params). Without this the tab would keep showing the old
  // selection while the model had quietly changed the dimensions underneath.
  const [tab, setTab] = useState(() =>
    orientationFor(width, height, allowed),
  );
  const [step, setStep] = useState(NATIVE_STEP);

  // Dimensions we just wrote ourselves. Without this, typing a width in Custom
  // fires the sync effect below, which re-derives the tab from the new pair and
  // throws the user back out of Custom -- the control fought the person using it.
  const selfWriteRef = useRef(null);

  // What the user is currently typing, per side. null means "show the live value".
  //
  // The fields were previously controlled straight off width/height and snapped on
  // every keystroke, which made them impossible to use: typing 5 became 256
  // immediately, so the next keystroke appended to that. Selecting the content and
  // typing produced an empty string, which was rejected, leaving state untouched --
  // so React reverted the field and nothing could be edited at all. A draft accepts
  // anything, including transitory values below the floor, and commits on blur or
  // Enter, which is when snapping to a multiple of 16 is meaningful.
  const [draft, setDraft] = useState({ width: null, height: null });
  const [lockRatio, setLockRatio] = useState(false);

  const w = Number(width);
  const h = Number(height);

  useEffect(() => {
    const key = `${width}x${height}`;
    setDraft({ width: null, height: null });
    if (selfWriteRef.current === key) {
      selfWriteRef.current = null;
      return;
    }
    // A model switch can leave the current size unreachable -- Qwen cannot hold
    // a 1280x1280 tile, and a legacy engine cannot be offered 1152x896. Rather
    // than show a control in a state it cannot represent, move to the nearest
    // size this model does allow.
    const reachable = allowed.some((p) => p.width === w && p.height === h);
    if (!reachable) {
      const near = nearestAllowed(w, h, allowed);
      if (near) {
        selfWriteRef.current = `${near.width}x${near.height}`;
        setWidth(near.width);
        setHeight(near.height);
        return;
      }
    }
    setTab(orientationFor(w, h, allowed));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [width, height, allowedKey]);

  function push(nextW, nextH) {
    const budgeted = fitToBudget(nextW, nextH, maxPixels);
    if (budgeted.width !== w || budgeted.height !== h) {
      selfWriteRef.current = `${budgeted.width}x${budgeted.height}`;
    }
    setWidth(budgeted.width);
    setHeight(budgeted.height);
  }

  function choosePreset(preset) {
    selfWriteRef.current = `${preset.width}x${preset.height}`;
    setTab(orientationOf(preset.width, preset.height));
    push(preset.width, preset.height);
  }

  /**
   * Commit one side: snap it to the step, derive the other if the ratio is
   * locked, and push. Values below the floor are raised here rather than while
   * typing.
   */
  function setSide(side, raw) {
    const value = Number(raw);
    if (!Number.isFinite(value) || value <= 0) return;
    const next = snap(value, step);
    let nw = side === "width" ? next : w;
    let nh = side === "height" ? next : h;
    if (lockRatio) {
      const other = side === "width" ? h : w;
      if (other > 0) {
        const scaled = side === "width"
          ? (next * other) / w
          : (next * w) / h;
        if (side === "width") nh = snap(scaled, step);
        else nw = snap(scaled, step);
      }
    }
    if (nw === w && nh === h) return;
    push(nw, nh);
  }

  /** Apply whatever is in the draft for one side, then hand the field back to state. */
  function commit(side) {
    const raw = draft[side];
    setDraft((d) => ({ ...d, [side]: null }));
    if (raw === null || String(raw).trim() === "") return;
    const value = Number(raw);
    // A non-number or a clear-to-empty is a cancellation, not an error: keep the
    // committed value rather than snapping from NaN.
    if (!Number.isFinite(value) || value <= 0) return;
    setSide(side, value);
  }

  function onTab(next) {
    setTab(next);
    if (next === "custom") return;
    // Switching tab should land on a real size for that tab, not leave the old
    // pair sitting under a tab that does not contain it.
    const list = presetsFor(next, modelInfo);
    if (!list.some((p) => p.width === w && p.height === h)) {
      const near = nearestAllowed(w, h, list);
      if (near) {
        selfWriteRef.current = `${near.width}x${near.height}`;
        push(near.width, near.height);
      }
    }
  }

  const list = presetsFor(tab, modelInfo);

  return (
    <div className="size-selector">
      <div className="size-selector-tabs" role="tablist">
        {ORIENTATIONS.map((o) => (
          <button
            key={o.id}
            type="button"
            role="tab"
            aria-selected={tab === o.id}
            className={`size-selector-tab${tab === o.id ? " is-active" : ""}`}
            onClick={() => onTab(o.id)}
          >
            {t(o.labelKey)}
          </button>
        ))}
      </div>

      {overAdvisory(w, h, modelInfo) && (
        <p className="size-advisory">
          {t("params.size.pixelAdvisory")} {Math.round(w * h).toLocaleString()} px
          {" → "}
          {t("params.size.pixelAdvisoryCap")}{" "}
          {advisoryFor(modelInfo).toLocaleString()} px
        </p>
      )}

      {tab === "custom" ? (
        <div className="size-selector-custom">
          <div className="size-selector-fields">
            <label className="size-field">
              <span className="size-field-label">{t("params.size.widthLabel")}</span>
              <input
                type="number"
                inputMode="numeric"
                min={256}
                step={step}
                value={draft.width ?? (Number.isFinite(w) ? w : "")}
                onChange={(e) => setDraft((d) => ({ ...d, width: e.target.value }))}
                onBlur={() => commit("width")}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    commit("width");
                    e.currentTarget.blur();
                  }
                }}
              />
            </label>
            <button
              type="button"
              className={`size-lock${lockRatio ? " is-on" : ""}`}
              aria-pressed={lockRatio}
              title={t("params.size.lockRatio")}
              onClick={() => setLockRatio((v) => !v)}
            >
              {lockRatio ? "\u29C9" : "\u29C5"}
            </button>
            <label className="size-field">
              <span className="size-field-label">{t("params.size.heightLabel")}</span>
              <input
                type="number"
                inputMode="numeric"
                min={256}
                step={step}
                value={draft.height ?? (Number.isFinite(h) ? h : "")}
                onChange={(e) => setDraft((d) => ({ ...d, height: e.target.value }))}
                onBlur={() => commit("height")}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    commit("height");
                    e.currentTarget.blur();
                  }
                }}
              />
            </label>
          </div>

          <div className="size-selector-steps" role="group">
            <span className="size-field-label">{t("params.size.step")}</span>
            {[NATIVE_STEP, MULTIPLE].map((s) => (
              <button
                key={s}
                type="button"
                className={`size-step${step === s ? " is-active" : ""}`}
                onClick={() => setStep(s)}
              >
                {s}
              </button>
            ))}
            {ratioLabel(w, h) && (
              <span className="size-selector-hint">
                {t("params.size.trueRatio")} {ratioLabel(w, h)}
              </span>
            )}
          </div>
        </div>
      ) : (
        <div className="size-presets">
          {list.map((p) => {
            const active = p.width === w && p.height === h;
            return (
              <button
                key={p.id}
                type="button"
                className={`size-preset${active ? " is-active" : ""}${
                  p.tier === "compact" ? " is-compact" : ""
                }`}
                aria-pressed={active}
                onClick={() => choosePreset(p)}
              >
                <span className="size-preset-dims">
                  {p.width}&times;{p.height}
                </span>
                <span className="size-preset-meta">
                  {t(p.nameKey)} &middot; {ratioLabel(p.width, p.height)}
                </span>
              </button>
            );
          })}
          {list.length === 0 && (
            <p className="size-selector-empty">{t("params.size.noPresets")}</p>
          )}
        </div>
      )}

    </div>
  );
}

// Closest allowed preset to a pair, by squared log-distance on the ratio so that
// orientation and scale are weighed together. Falls back to the first entry.
function nearestAllowed(width, height, list) {
  if (!list || list.length === 0) return null;
  const target = Math.log((Number(width) || 1) / (Number(height) || 1));
  let best = null;
  let bestCost = Infinity;
  for (const p of list) {
    const d = Math.log(p.width / p.height) - target;
    const scale = Math.log((p.width * p.height) / ((Number(width) || 1) * (Number(height) || 1)));
    const cost = d * d + scale * scale;
    if (cost < bestCost) {
      bestCost = cost;
      best = p;
    }
  }
  return best;
}