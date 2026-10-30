export function clampPage(value, pages) {
  if (typeof value === "string" && value.trim() === "") return null;
  const max = Math.max(1, Math.trunc(Number(pages) || 1));
  const page = Math.trunc(Number(value));
  if (!Number.isFinite(page)) return null;
  return Math.min(max, Math.max(1, page));
}

export function shouldNavigatePage(event, { blocked = false } = {}) {
  if (blocked) return false;
  if (
    event.defaultPrevented ||
    event.metaKey ||
    event.ctrlKey ||
    event.altKey ||
    event.shiftKey
  ) {
    return false;
  }
  const target = event.target;
  if (
    target &&
    (target.isContentEditable ||
      ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName))
  ) {
    return false;
  }
  return event.key === "ArrowLeft" || event.key === "ArrowRight";
}
