/**
 * Clipboard writes that either work or tell you why they did not.
 *
 * `navigator.clipboard` is only available in a SECURE context. localhost counts as
 * one, but the same app opened from another machine over the LAN does not -- and then
 * every copy button silently does nothing, because the rejection was swallowed by an
 * empty `catch`. That is the "the button does nothing" report: no error, no change,
 * no hint.
 *
 * So: try the modern API, fall back to the legacy textarea trick, and if BOTH fail
 * surface the failure instead of eating it. Returns a boolean so callers can show
 * their own "copied" state only when something was actually copied.
 */

async function legacyCopy(text) {
  if (typeof document === "undefined") return false;
  const area = document.createElement("textarea");
  area.value = text;
  // Off-screen rather than display:none: a hidden element cannot be selected, and
  // execCommand refuses to copy an unselectable one.
  area.setAttribute("readonly", "");
  area.style.position = "fixed";
  area.style.top = "-1000px";
  area.style.opacity = "0";
  document.body.appendChild(area);
  try {
    area.select();
    area.setSelectionRange(0, area.value.length);
    return typeof document.execCommand === "function" && document.execCommand("copy");
  } catch {
    return false;
  } finally {
    document.body.removeChild(area);
  }
}

export async function copyText(text) {
  const value = String(text ?? "");
  if (!value) return false;
  try {
    if (navigator?.clipboard?.writeText) {
      await navigator.clipboard.writeText(value);
      return true;
    }
  } catch {
    // Fall through to the legacy path rather than giving up here.
  }
  return legacyCopy(value);
}

/**
 * Copy an image blob. There is no legacy fallback worth having for binary data, so
 * this reports failure plainly -- the caller must show it, because silently doing
 * nothing is the bug this module exists to stop.
 */
export async function copyImageBlob(blob) {
  if (!blob) return false;
  if (!navigator?.clipboard?.write || typeof ClipboardItem === "undefined") {
    return false;
  }
  try {
    await navigator.clipboard.write([new ClipboardItem({ [blob.type || "image/png"]: blob })]);
    return true;
  } catch {
    return false;
  }
}

export default copyText;