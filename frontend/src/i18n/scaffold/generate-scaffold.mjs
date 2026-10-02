/**
 * Regenerate the empty translation scaffolds in this directory.
 *
 *   node src/i18n/scaffold/generate-scaffold.mjs
 *
 * Run it after adding or renaming a key in ../parts/*.js or ../strings.js. The
 * scaffold is the translator's work order: one line per key, in catalogue order,
 * with the English source alongside so nothing has to be looked up.
 *
 * WHY THE VALUES ARE null AND NOT ""
 * translate.js resolves a string with `entry[lang] ?? entry.en ?? key`. The `??`
 * operator falls back on null and undefined ONLY. An empty string is a real value,
 * so a translator who fills a line with "" ships a blank label -- exactly the
 * failure this scaffolding exists to make impossible. null means "not translated
 * yet" and renders as English, which is honest and harmless.
 *
 * WHY THESE FILES ARE NOT IN ../parts/
 * ../parts/*.js is globbed by strings.js and by test_i18n.py, and both assume the
 * directory holds DISJOINT catalogues of the same four shipped languages. A second
 * catalogue of the same keys would be read as a duplicate-key collision. These files
 * are inert work orders: nothing imports them until a language's status in
 * ../languages.js flips from "scaffold" to "shipped".
 */
import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { STRINGS } from "../strings.js";
import { SCAFFOLD_LANGUAGES } from "../languages.js";

const HERE = dirname(fileURLToPath(import.meta.url));

/** Make an English string safe to sit in a trailing line comment. */
function asComment(value) {
  return String(value).replace(/\*\//g, "*\\/").replace(/\n/g, "\\n");
}

/** "app.generate" -> "appGenerate", for a camelCase export name. */
function exportName(code) {
  return code + "Strings";
}

for (const lang of SCAFFOLD_LANGUAGES) {
  const keys = Object.keys(STRINGS);
  const lines = keys.map((key) => {
    const source = STRINGS[key]?.en ?? "";
    return `  ${JSON.stringify(key)}: null, // ${asComment(source)}`;
  });

  const header = `// ${lang.label} translation scaffold -- GENERATED, do not hand-edit the key list.
// Regenerate: node src/i18n/scaffold/generate-scaffold.mjs
//
// Every one of the ${keys.length} interface strings, with the English source as a comment.
// Replace null with the ${lang.label} translation. Leave a key null if you are unsure:
// null renders as English, whereas "" renders as a blank label (see the header note in
// generate-scaffold.mjs about why the fallback uses ?? and not ||).
//
// Native terms to keep in English unless there is a settled ${lang.label} equivalent:
// FLUX, Krea, Z-Image, LoRA, SDXL, MLX, CoreML, Metal, mflux, safetensors, VAE, UNet,
// sampler names, and the licence identifiers in src/data/licences.js.

export const ${exportName(lang.code)} = {
`;

  const path = join(HERE, `${lang.code}.js`);
  writeFileSync(path, header + lines.join("\n") + "\n};\n", "utf8");
  console.log(`  wrote ${lang.code}.js  (${keys.length} keys, ${lang.label})`);
}

console.log(`  ${SCAFFOLD_LANGUAGES.length} scaffolds regenerated`);