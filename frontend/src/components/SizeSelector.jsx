import { useEffect, useMemo, useRef, useState } from "react";
import { useI18n } from "../i18n/I18nContext";

/**
 * Resolution and aspect-ratio selector.
 *
 * Replaces a 25-entry dropdown that mixed exact pixel resolutions with aspect
 * ratios, plus per-model preset buttons. Two orthogonal decisions are now made
 * separately, which is the point: people choose a shape first and a scale second,
 * not a single fused "1024x1024 (1:1)" string.
 *
 *   base   256 / 512 / 768 / 1024  -- how much detail
 *   ratio  1:1 16:9 3:2 9:16 2:3   -- what shape
 *
 * Custom mode drops the ratio row and exposes width/height directly for the cases
 * the grid cannot express (cinemascope, an exact reference size).
 *
 * Every emitted dimension is a multiple of 16 with a floor of 256. That is not
 * cosmetic: mflux rejects a dimension that is not a multiple of 16, and it is the
 * same rule the existing max_pixels clamp already enforced.
 */

const BASES = [256, 512, 768, 1024];

const RATIOS = [
  { id: "1:1", w: 1, h: 1 },
  { id: "4:3", w: 4, h: 3 },
  { id: "3:2", w: 3, h: 2 },
  { id: "16:9", w: 16, h: 9 },
  { id: "21:9", w: 21, h: 9 },
  { id: "3:4", w: 3, h: 4 },
  { id: "2:3", w: 2, h: 3 },
  { id: "9:16", w: 9, h: 16 },
  { id: "9:21", w: 9, h: 21 },
];

const MULTIPLE = 16;
const FLOOR = 256;

function snap(value) {
  const n = Math.max(FLOOR, Math.round(Number(value) / MULTIPLE) * MULTIPLE);
  return n;
}

/**
 * Final pixel dimensions for a base and a ratio, base applied to the long edge.
 *
 * The floor is applied by scaling the WHOLE pair, not by clamping the short edge.
 * Clamping alone is what a first pass did, and it silently destroyed the ratio: at
 * base 256 the 16:9 short edge computes to 144, which the 256 floor raised to 256,
 * turning every ratio at that base into a square. Scaling keeps the user's chosen
 * shape, which is the entire reason this control exists.
 */
function dimensionsFor(base, ratio) {
  const { w, h } = ratio;
  let width, height;
  if (w >= h) {
    width = base;
    height = (base * h) / w;
  } else {
    width = (base * w) / h;
    height = base;
  }
  const short = Math.min(width, height);
  if (short < FLOOR) {
    const grow = FLOOR / short;
    width *= grow;
    height *= grow;
  }
  return { width: snap(width), height: snap(height) };
}

/**
 * Recover base+ratio from a width/height pair.
 *
 * Needed whenever the size is set from outside this component: a hydrated
 * request, "reuse params", or a model switch that resets the dimensions. Returning
 * null means the pair is not on the grid, and the component falls back to Custom
 * rather than silently snapping the user's size to something they did not ask for.
 */
function derive(width, height) {
  const w = Number(width);
  const h = Number(height);
  if (!Number.isFinite(w) || !Number.isFinite(h) || w <= 0 || h <= 0) return null;

  const long = Math.max(w, h);

  // The base is the long edge, snapped to the nearest offered step.
  let best = null;
  let bestDelta = Infinity;
  for (const b of BASES) {
    const delta = Math.abs(long - b);
    if (delta < bestDelta) {
      bestDelta = delta;
      best = b;
    }
  }
  // Only claim a grid hit when the long edge is close to a real step; otherwise a
  // 700px image would be reported as "768" and quietly change size.
  if (!best || bestDelta > 32) return null;

  for (const ratio of RATIOS) {
    const d = dimensionsFor(best, ratio);
    if (Math.abs(d.width - w) <= 16 && Math.abs(d.height - h) <= 16) {
      return { base: best, ratioId: ratio.id };
    }
  }
  return null;
}

/**
 * Scale a pair down until it fits a pixel budget, preserving shape.
 * Mirrors the existing max_pixels behaviour so the two paths cannot disagree.
 */
function fitToBudget(width, height, maxPixels) {
  const cap = Number(maxPixels);
  // maxPixels arrives as null for any model without a pixel cap, and Number(null)
  // is 0 -- finite, and below every real size. Treating that as a budget produced a
  // permanent red "Over 0 px -- generation may fail" under a perfectly normal size.
  if (maxPixels == null || maxPixels === "" || !Number.isFinite(cap) || cap <= 0) {
    return { width, height };
  }
  if (width * height <= cap) return { width, height };
  const scale = Math.sqrt(cap / (width * height));
  return { width: snap(width * scale), height: snap(height * scale) };
}

export default function SizeSelector({
  width,
  setWidth,
  height,
  setHeight,
  maxPixels,
  modelInfo,
}) {
  const { t } = useI18n();

  // The derived grid position, kept in step with externally-driven width/height.
  const derived = useMemo(() => derive(width, height), [width, height]);
  const [mode, setMode] = useState("standard");
  // Dimensions we just wrote ourselves. Without this, typing a width in Custom mode
  // fires the sync effect below, which re-derives base+ratio from the new pair and
  // throws the user back out of Custom -- the control fought the person using it.
  const selfWriteRef = useRef(null);
  // Keep-ratio: when on, editing one side derives the other from the current shape.
  const [lockRatio, setLockRatio] = useState(false);
  const [base, setBase] = useState(derived ? derived.base : 1024);
  const [ratioId, setRatioId] = useState(derived ? derived.ratioId : "1:1");

  // Adopt the incoming size whenever it comes from outside (hydration, model
  // switch, reuse params). Without this the buttons would keep showing the old
  // selection while the model had quietly changed the dimensions underneath.
  useEffect(() => {
    const key = `${width}x${height}`;
    if (selfWriteRef.current === key) {
      selfWriteRef.current = null;
      return;
    }
    const d = derive(width, height);
    if (d) {
      setBase(d.base);
      setRatioId(d.ratioId);
      setMode("standard");
    } else {
      setMode("custom");
    }
    // Only on the dimensions themselves: re-running on our own writes would
    // fight the button press.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [width, height]);

  function apply(nextBase, nextRatioId) {
    const target = RATIOS.find((r) => r.id === nextRatioId) ?? RATIOS[0];
    const raw = dimensionsFor(nextBase, target);
    const d = fitToBudget(raw.width, raw.height, maxPixels);
    if (d.width !== width || d.height !== height) selfWriteRef.current = `${d.width}x${d.height}`;
    setWidth(d.width);
    setHeight(d.height);
  }

  /**
   * One side changed in Custom mode.
   *
   * With the ratio locked the other side follows, preserving the shape the user is
   * looking at. Either way the pair is marked as self-written so the effect that
   * re-derives base+ratio does not interpret a keystroke as an external size change
   * and bounce the control back to the grid.
   */
  function setSide(side, raw) {
    const value = Number(raw);
    if (!Number.isFinite(value) || value <= 0) return;
    const next = snap(value);
    let w = side === "width" ? next : Number(width);
    let h = side === "height" ? next : Number(height);
    if (lockRatio) {
      const other = side === "width" ? Number(height) : Number(width);
      if (other > 0) {
        const scaled = side === "width"
          ? (next * other) / Number(width)
          : (next * Number(width)) / Number(height);
        if (side === "width") h = snap(scaled);
        else w = snap(scaled);
      }
    }
    if (w === Number(width) && h === Number(height)) return;
    selfWriteRef.current = `${w}x${h}`;
    setWidth(w);
    setHeight(h);
  }

  function selectBase(next) {
    setBase(next);
    if (mode === "standard") apply(next, ratioId);
  }

  function selectRatio(nextId) {
    setRatioId(nextId);
    if (mode === "standard") apply(base, nextId);
  }

  const preview = fitToBudget(width, height, maxPixels);
  const budget =
    maxPixels == null || maxPixels === "" ? null : Number(maxPixels);
  const hasBudget = Number.isFinite(budget) && budget > 0;
  const overBudget = hasBudget && Number(width) * Number(height) > budget;

  return (
    <div className="size-selector">
      <div className="size-selector-head">
        <span className="size-selector-label">{t("params.size.label")}</span>
        <span className="size-selector-readout">
          {preview.width} × {preview.height}
        </span>
      </div>

      <div className="size-base-row" role="group" aria-label={t("params.size.baseLabel")}>
        {BASES.map((b) => (
          <button
            key={b}
            type="button"
            className={`size-base-btn${mode === "standard" && base === b ? " active" : ""}`}
            aria-pressed={mode === "standard" && base === b}
            onClick={() => selectBase(b)}
            title={t("params.size.baseTitle", { base: b })}
          >
            {b}
          </button>
        ))}
        <button
          type="button"
          className={`size-base-btn size-custom-btn${mode === "custom" ? " active" : ""}`}
          aria-pressed={mode === "custom"}
          onClick={() => setMode("custom")}
          title={t("params.size.customTitle")}
        >
          {t("params.size.customShort")}
        </button>
      </div>
      {mode === "standard" ? (
        <div className="size-ratio-row" role="group" aria-label={t("params.size.ratioLabel")}>
          {RATIOS.map((r) => (
            <button
              key={r.id}
              type="button"
              className={`size-ratio-btn${ratioId === r.id ? " active" : ""}`}
              aria-pressed={ratioId === r.id}
              onClick={() => selectRatio(r.id)}
              title={t("params.size.ratioTitle", { ratio: r.id })}
            >
              <span
                className="size-ratio-glyph"
                style={{ aspectRatio: `${r.w} / ${r.h}` }}
                aria-hidden="true"
              />
              <span className="size-ratio-text">{r.id}</span>
            </button>
          ))}
        </div>
      ) : (
        <>
          <div className="size-custom-row">
            <label className="size-custom-field">
              <span>{t("params.size.widthLabel")}</span>
              <input
                type="number"
                min={FLOOR}
                step={MULTIPLE}
                value={width}
                onChange={(e) => setSide("width", e.target.value)}
              />
            </label>
            <label className="size-custom-field">
              <span>{t("params.size.heightLabel")}</span>
              <input
                type="number"
                min={FLOOR}
                step={MULTIPLE}
                value={height}
                onChange={(e) => setSide("height", e.target.value)}
              />
            </label>
            <button
              type="button"
              className={`btn-mini size-lock-btn${lockRatio ? " active" : ""}`}
              aria-pressed={lockRatio}
              onClick={() => setLockRatio((v) => !v)}
              title={lockRatio ? t("params.size.unlockRatioTitle") : t("params.size.lockRatioTitle")}
            >
              {t("params.size.lockRatio")}
            </button>
            <button
              type="button"
              className="btn-mini"
              onClick={() => {
                setMode("standard");
                apply(base, ratioId);
              }}
              title={t("params.size.backToGridTitle")}
            >
              {t("params.size.backToGrid")}
            </button>
          </div>
          {lockRatio && (
            <p className="size-lock-hint" role="status">
              {t("params.size.lockRatioHint")}
            </p>
          )}
        </>
      )}

      {overBudget && (
        <p className="size-budget-warning" role="status">
          {t("params.size.budgetWarning", { max: budget })}
        </p>
      )}

      {modelInfo?.max_pixels ? (
        <p className="size-budget-hint">{t("params.size.budgetHint")}</p>
      ) : null}
    </div>
  );
}