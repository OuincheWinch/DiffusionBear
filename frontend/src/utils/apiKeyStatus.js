// Mirrors backend authenticate(): weak env/legacy keys are ignored, so they
// must not count as a key the browser can already use.
export function hasUsableApiKey(status) {
  return Boolean(
    status?.environment?.strong ||
      status?.legacy?.strong ||
      status?.keys?.length,
  );
}
