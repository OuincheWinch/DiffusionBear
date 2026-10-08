// Render smoke test for SizeSelector.
//
// This exists because of a specific, shipped failure. A temporal dead zone in
// this component -- canStart reading showBar above its own declaration -- threw
// on every render and blanked the app's interface recovery boundary. Lint was
// clean, 504 backend tests passed and CI was green, because none of them ever
// evaluated a React component. Only a human opening the installed app found it.
//
// So: actually render it. A static TDZ scanner was tried first and rejected --
// it flagged 63 false positives, because it cannot tell a function parameter
// from a declaration in another scope. Rendering has no such problem.
//
// Aliased away: the real I18nProvider, which reaches for the network on mount.
// The stub returns the key it was asked for, which also proves WHICH keys the
// component requests -- a typo in a labelKey shows up as a suspicious string in
// the output rather than as an empty button nobody notices.

import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import SizeSelector from "/SIZE_SELECTOR_PATH";
import ModelInstaller from "/MODEL_INSTALLER_PATH";

export function useI18n() {
  return { t: (key) => `«${key}»`, language: "en" };
}

export const cases = [];

function record(name, fn) {
  try {
    cases.push({ name, ok: true, html: fn() });
  } catch (err) {
    cases.push({ name, ok: false, error: String((err && err.message) || err) });
  }
}

const base = {
  width: 1024,
  setWidth() {},
  height: 1024,
  setHeight() {},
  maxPixels: null,
};

record("square preset selected", () =>
  renderToStaticMarkup(createElement(SizeSelector, { ...base })),
);

record("registry default 512x768", () =>
  renderToStaticMarkup(
    createElement(SizeSelector, { ...base, width: 512, height: 768 }),
  ),
);

record("portrait preset", () =>
  renderToStaticMarkup(createElement(SizeSelector, { ...base, width: 896, height: 1152 })),
);

record("qwen 1280x1280 over its comfortable budget", () =>
  renderToStaticMarkup(
    createElement(SizeSelector, {
      ...base,
      width: 1280,
      height: 1280,
      maxPixels: 1024 * 768,
      modelInfo: { ecosystem: "Qwen-Image 2.1" },
    }),
  ),
);

record("legacy engine sees the small sizes", () =>
  renderToStaticMarkup(
    createElement(SizeSelector, {
      ...base,
      modelInfo: { ecosystem: "SD15", legacy_sizes: true },
    }),
  ),
);

record("modelInfo absent entirely", () =>
  renderToStaticMarkup(createElement(SizeSelector, { ...base, modelInfo: undefined })),
);

record("non-numeric dimensions", () =>
  renderToStaticMarkup(
    createElement(SizeSelector, { ...base, width: NaN, height: undefined }),
  ),
);

// ModelInstaller is here for a specific reason: its temporal dead zone shipped in
// v0.3.4 and blanked the app. canStart read showBar three lines above showBar's own
// declaration, so every render threw. The props below are the minimum that gets
// execution PAST the earlier reads (modelInfo.download_repo) so that the dead zone
// is what actually fires -- otherwise the harness fails on a missing prop and
// proves nothing about the bug it exists to catch.
const miProps = {
  modelInfo: {
    id: "flux2-klein-4b",
    label: "FLUX.2-klein 4B",
    download_repo: "mlx-community/flux2-klein-4b-4bit",
    installed: false,
  },
  task: null,
};

record("ModelInstaller idle", () =>
  renderToStaticMarkup(createElement(ModelInstaller, { ...miProps })),
);

record("ModelInstaller with a cancelled task", () =>
  renderToStaticMarkup(
    createElement(ModelInstaller, {
      ...miProps,
      task: { status: "cancelled", total_bytes: 100, downloaded_bytes: 40 },
    }),
  ),
);

console.log(JSON.stringify(cases));
