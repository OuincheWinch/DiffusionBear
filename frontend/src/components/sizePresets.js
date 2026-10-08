// Native resolution presets -- the DATA layer for the size selector.
//
// Deliberately free of React, i18n and the DOM so it can be executed directly
// under node by backend/test_size_selector.py. The tests exercise this file
// rather than a Python re-implementation, because a re-implementation tests
// the copy instead of the shipped code.
//
// WHY PRESETS AND NOT base x ratio
// The old control multiplied a base (256/512/768/1024) by a ratio and snapped.
// For 16:9 at base 1024 that yields 1024x576, which is not a resolution any of
// these models was trained on, and the result is soft, warped anatomy. The
// numbers below are the buckets the models actually know: applying a ratio to a
// base is the defect, so the ratio now follows the resolution, never the reverse.
//
// TO ADD A RESOLUTION: append one object to the relevant list. Nothing else in
// the app needs to change -- the view reads these lists, the tests validate
// every entry, and the i18n name is looked up from `nameKey`.

// mflux does not reject a dimension that is not a multiple of 16. It logs
// "Width and height should be multiples of 16. Rounding down." and floors it
// (mflux/models/common/config/config.py:41). That is worse than a rejection:
// the run succeeds at a size nobody asked for. So we snap on the way in.
export const MULTIPLE = 16;

// Every native preset sits on a 64px grid. Not a backend requirement -- 16 is
// the requirement -- but the VAE patch grid and these models' training
// buckets both land on 64, so it is the right default for hand-entered sizes.
export const NATIVE_STEP = 64;

// Nothing may be generated below this on any engine.
export const FLOOR = 256;

export const ORIENTATIONS = [
  { id: "square", labelKey: "params.size.orientation.square" },
  { id: "landscape", labelKey: "params.size.orientation.landscape" },
  { id: "portrait", labelKey: "params.size.orientation.portrait" },
  { id: "custom", labelKey: "params.size.orientation.custom" },
];

// `ratio` is deliberately NOT stored. It is derived from width/height by
// ratioLabel() so the label can never drift from the pixels it describes.
// `nameKey` is the friendly name; several of these buckets only approximate the
// ratio people ask for, so the true reduced ratio is shown next to the name.
// Tier "compact" -- the small, classic base resolutions. They are simply offered,
// because they are sizes people ask for. "Compact" is a label, not a warning.
//
// A previous revision hid anything under 768 px behind a checkbox. It was removed
// because it cost a line of vertical space to hide exactly one button (512x512)
// and did nothing at all on the Landscape and Portrait tabs, and because silently
// removing a size someone can see is not a decision the app should make for them.
export const SIZE_PRESETS = {
  square: [
    { id: "sq-512", width: 512, height: 512, nameKey: "params.size.name.compact", tier: "compact" },
    { id: "sq-768", width: 768, height: 768, nameKey: "params.size.name.compact", tier: "compact" },
    { id: "sq-1024", width: 1024, height: 1024, nameKey: "params.size.name.standard", tier: "native" },
    { id: "sq-1280", width: 1280, height: 1280, nameKey: "params.size.name.large", tier: "native" },
  ],
  landscape: [
    { id: "ls-768x512", width: 768, height: 512, nameKey: "params.size.name.compact", tier: "compact" },
    { id: "ls-1024x768", width: 1024, height: 768, nameKey: "params.size.name.standard", tier: "native" },
    { id: "ls-1152x896", width: 1152, height: 896, nameKey: "params.size.name.standard", tier: "native" },
    { id: "ls-1344x768", width: 1344, height: 768, nameKey: "params.size.name.wide", tier: "native" },
    { id: "ls-1216x832", width: 1216, height: 832, nameKey: "params.size.name.photo", tier: "native" },
    { id: "ls-1536x640", width: 1536, height: 640, nameKey: "params.size.name.cinematic", tier: "native" },
  ],
  portrait: [
    { id: "pt-512x768", width: 512, height: 768, nameKey: "params.size.name.compact", tier: "compact" },
    { id: "pt-768x1024", width: 768, height: 1024, nameKey: "params.size.name.standard", tier: "native" },
    { id: "pt-896x1152", width: 896, height: 1152, nameKey: "params.size.name.standard", tier: "native" },
    { id: "pt-1024x1280", width: 1024, height: 1280, nameKey: "params.size.name.social", tier: "native" },
    { id: "pt-768x1344", width: 768, height: 1344, nameKey: "params.size.name.tall", tier: "native" },
    { id: "pt-832x1216", width: 832, height: 1216, nameKey: "params.size.name.photoPortrait", tier: "native" },
  ],
};

// Tier "legacy" -- offered only to an engine that declares `legacy_sizes: true`.
// No model in the registry does, so these are currently unreachable; they exist so
// that adding an SD1.5-class engine is a one-flag change rather than a UI change.
export const LEGACY_PRESETS = [
  { id: "lg-512", orientation: "square", width: 512, height: 512, nameKey: "params.size.name.legacy512", tier: "legacy" },
  { id: "lg-256", orientation: "square", width: 256, height: 256, nameKey: "params.size.name.legacy256", tier: "legacy" },
];

// Per-ecosystem capability. `modern` is what switches the size floor on.
//
// WHY THE FLOOR IS THE LONG EDGE, AND WHY IT IS 768
// A rule of "nothing under 768" cannot be applied to the short edge: every
// model in the registry -- all nine -- defaults to 512x768, and 512x768 is a
// size we have measured working on krea2 and qwen (see AGENTS.md; the shape
// that OOMs is 1024x688 plus a reference image, not 512x768). Rejecting it
// would leave the control unable to show the size the app is actually running
// at, which is its own defect.
//
// Applied to the LONG edge instead, 768 is exactly right and needs no fudge:
// 512x768 clears it (long edge 768) and stays available, while 512x512 and
// 256x256 do not and are withheld as the visual noise they are. A square and a
// 512x768 landscape are not the same amount of signal, and the long edge is
// what tells them apart.
export const ECOSYSTEMS = {
  "FLUX.2": { modern: true },
  SDXL: { modern: true },
  "Krea 2": { modern: true },
  ZImageTurbo: { modern: true },
  "Qwen-Image 2.1": { modern: true },
};

// An ADVISORY pixel ceiling, per ecosystem. Not a block and not a clamp: the
// control still offers the size, because whether it fits depends on the machine
// and only the owner knows that.
//
// It exists because Qwen-Image 2.1 at 1024x1024 is a confirmed out-of-memory on
// a 16GB M1 (Metal bf16 VAE decode, recorded in AGENTS.md), and this control now
// offers 1024x1024 and 1280x1280 one click away. The per-model pixel caps were
// removed by decision on 2026-09-22, so nothing in the pipeline stops it -- a
// silent OOM at generation time is the worst place to find out.
export const ADVISORY_PIXELS = {
  "Qwen-Image 2.1": 768 * 768,
};

// The advisory ceiling for this model, or null when there is none.
export function advisoryFor(modelInfo) {
  const eco = modelInfo?.ecosystem;
  if (!eco) return null;
  return ADVISORY_PIXELS[eco] ?? null;
}

// Whether this size is over its model's advisory ceiling. Advisory only: the
// caller is expected to warn, not to refuse.
export function overAdvisory(width, height, modelInfo) {
  const cap = advisoryFor(modelInfo);
  if (!cap) return false;
  if (!Number.isFinite(width) || !Number.isFinite(height)) return false;
  return width * height > cap;
}

// A model with no ecosystem recorded is treated as modern: every engine in the
// registry is, and defaulting to legacy would offer noise by default.
//
// Nothing is legacy TODAY, so the small sizes are unreachable until a registry
// entry opts in with `legacy_sizes: true`. That is deliberate: an unreachable
// branch is better than a button that offers 512x512 to a model that cannot use
// it, and it means adding an SD1.5-class engine is a one-flag change rather
// than a UI change.
export function isModern(modelInfo) {
  if (!modelInfo) return true;
  if (modelInfo.legacy_sizes === true) return false;
  const eco = modelInfo.ecosystem;
  if (!eco) return true;
  const caps = ECOSYSTEMS[eco];
  return caps ? caps.modern !== false : true;
}

// The presets for one orientation tab.
//
// showAll is TRUE BY DEFAULT, so every size is offered unless the user turns the
// guidance on. An earlier revision filtered the sub-768 sizes out automatically
// and that was wrong: the small resolutions are ones people actually ask for,
// and a guideline about quality is not grounds for removing a choice. The gate
// still exists, because it is useful, but it is now something the user puts on
// rather than something imposed on them.
export function presetsFor(orientation, modelInfo) {
  if (orientation === "custom") return [];
  const native = SIZE_PRESETS[orientation] || [];
  // The legacy tier only exists for an engine that declares itself legacy. Nothing
  // in the registry does today, which is deliberate: an unreachable branch is
  // better than a button offering a size the model cannot use, and adding an
  // SD1.5-class engine is then a one-flag change instead of a UI change.
  if (!modelInfo || isModern(modelInfo)) return native;
  // Deduped on dimensions, not id: lg-512 is also 512x512, and both would match
  // `active` and render as two identical highlighted tiles.
  const seen = new Set(native.map((p) => `${p.width}x${p.height}`));
  const extra = LEGACY_PRESETS.filter(
    (p) => p.orientation === orientation && !seen.has(`${p.width}x${p.height}`),
  );
  return [...native, ...extra];
}

// Every preset this model may use, across all tabs. The view uses it to detect a
// size that is no longer reachable (e.g. after a model switch) and to pick a
// sensible neighbour instead of leaving the control in an impossible state.
export function allAllowed(modelInfo) {
  return ORIENTATIONS.filter((o) => o.id !== "custom").flatMap((o) =>
    presetsFor(o.id, modelInfo),
  );
}

// Which tab owns this size, or "custom" if it is not a preset at all.
export function orientationFor(width, height, presets) {
  const match = presets.find((p) => p.width === width && p.height === height);
  if (match) return match.orientation || orientationOf(width, height);
  return "custom";
}

// Classify by shape. Used both when importing an external size and when
// validating that a preset sits in the tab it claims to belong to. A size that is
// not a finite pair of numbers has no shape; null says so rather than guessing,
// because guessing sent a NaN width to the "portrait" branch.
export function orientationOf(width, height) {
  if (!Number.isFinite(width) || !Number.isFinite(height)) return null;
  if (width === height) return "square";
  return width > height ? "landscape" : "portrait";
}

function gcd(a, b) {
  a = Math.abs(a);
  b = Math.abs(b);
  while (b) {
    const t = b;
    b = a % b;
    a = t;
  }
  return a || 1;
}

// The TRUE reduced ratio, e.g. 1216x832 -> "19:13". Derived from the pixels so
// the displayed ratio is always the ratio actually being generated.
//
// Returns null for a non-finite pair. It used to return the string "NaN:NaN",
// which the Custom panel then printed next to the width and height fields -- a
// ratio displayed while the width field sat empty.
export function ratioLabel(width, height) {
  if (!Number.isFinite(width) || !Number.isFinite(height)) return null;
  const g = gcd(width, height);
  return `${width / g}:${height / g}`;
}

// Decimal ratio, for layout maths and tests.
export function ratioValue(width, height) {
  return width / height;
}

export function snap(value, step = MULTIPLE) {
  const n = Number(value);
  if (!Number.isFinite(n)) return null;
  return Math.max(FLOOR, Math.round(n / step) * step);
}

// Scale a pair down until it fits a pixel budget, preserving shape. Returns the
// pair unchanged when there is no budget or it already fits.
export function fitToBudget(width, height, maxPixels) {
  if (!maxPixels || width * height <= maxPixels) return { width, height };
  const scale = Math.sqrt(maxPixels / (width * height));
  return {
    width: Math.max(MULTIPLE, Math.floor((width * scale) / MULTIPLE) * MULTIPLE),
    height: Math.max(MULTIPLE, Math.floor((height * scale) / MULTIPLE) * MULTIPLE),
  };
}