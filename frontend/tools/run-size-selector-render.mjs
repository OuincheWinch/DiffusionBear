#!/usr/bin/env node
// Bundles and renders tools/size-selector-render.mjs, then prints one line per
// case. Driven by backend/test_size_selector_render.py; kept as a script so the
// vite plumbing stays out of the test file.
//
// The generated files live inside the project rather than in os.tmpdir() because
// a temp dir has no node_modules above it, and vite cannot resolve "react" from
// there. Everything written here is removed afterwards.

import { build } from "vite";
import { mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

// globalThis.process rather than process: this file is a Node script, but the linter
// is oxlint and applies the browser globals, where process is not defined.
const node = globalThis.process;
const component_arg = node.argv[3];
const root = resolve(node.argv[2] ?? ".");
// Any component may be pointed at, so the same harness can be used to prove it
// catches a known-bad revision rather than only ever agreeing with the current
// one: `node tools/run-size-selector-render.mjs . src/components/Foo.jsx`.
const component = resolve(root, component_arg ?? "src/components/SizeSelector.jsx");
const installer = resolve(root, "src/components/ModelInstaller.jsx");
const entry = resolve(root, "tools/size-selector-render.mjs");

const work = join(root, "tools/.render-smoke");
mkdirSync(work, { recursive: true });

// The real I18nProvider pulls in the API client, which is not what this test is
// about. A virtual module swaps just that import. It is a plugin rather than a
// resolve.alias because aliasing a path that does not exist makes the bundler
// emit it as an external bare specifier, which then fails to resolve at runtime.
// Returning a virtual id keeps it inlined and the test hermetic -- no network, no
// real translations, and the output shows which keys the component asks for.
const i18nStub = {
  name: "i18n-stub",
  enforce: "pre",
  resolveId(source) {
    return /i18n[\\/]I18nContext$/.test(source) ? "\0i18n-stub" : null;
  },
  load(id) {
    if (id !== "\0i18n-stub") return null;
    return "export function useI18n() {\n" +
      "  return { t: (k) => '\\u00ab' + k + '\\u00bb', language: 'en' };\n" +
      "}\n";
  },
};

try {
  const infile = join(work, "entry.mjs");
  writeFileSync(infile, readFileSync(entry, "utf8")
    .replace("/SIZE_SELECTOR_PATH", component)
    .replace("/MODEL_INSTALLER_PATH", installer));

  await build({
    root,
    logLevel: "error",
    configFile: false,
    plugins: [i18nStub],
    build: {
      ssr: infile,
      outDir: join(work, "out"),
      emptyOutDir: true,
      minify: false,
      rollupOptions: { output: { entryFileNames: "bundle.mjs" } },
    },
  });

  const mod = await import(pathToFileURL(join(work, "out", "bundle.mjs")).href);
  const cases = mod.cases;
  let failed = 0;
  for (const c of cases) {
    if (c.ok) {
      console.log(`PASS ${c.name} (${c.html.length} bytes)`);
    } else {
      failed += 1;
      console.log(`FAIL ${c.name}: ${c.error}`);
    }
  }
  console.log(`SUMMARY ${cases.length - failed}/${cases.length}`);
  node.exit(failed ? 1 : 0);
} finally {
  rmSync(work, { recursive: true, force: true });
}