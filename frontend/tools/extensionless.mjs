// Teach node the app's import style.
//
// The frontend imports without extensions -- `import { STRINGS } from "./strings"` --
// because that is what vite resolves and what the rest of the codebase does. Node's ESM
// resolver requires the extension, so anything that wants to READ the real catalogues
// from node (the i18n coverage tests, the translation build script) fails with
// ERR_MODULE_NOT_FOUND.
//
// This is not cosmetic. test_i18n.py shells out to node to compare a translation against
// the true key list; without this hook the call failed, the helper returned None, and the
// test SKIPPED -- while still reporting OK. A green skip is worse than a red failure.
export async function resolve(specifier, context, next) {
  try {
    return await next(specifier, context);
  } catch (error) {
    const relative = specifier.startsWith(".") || specifier.startsWith("/");
    const hasExtension = /\.[a-z]+$/i.test(specifier);
    if (relative && !hasExtension) {
      for (const candidate of [`${specifier}.js`, `${specifier}/index.js`]) {
        try {
          return await next(candidate, context);
        } catch {
          // keep trying
        }
      }
    }
    throw error;
  }
}
