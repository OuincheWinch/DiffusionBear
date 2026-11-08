const ERROR_CODE = /(?:^|:\s*)([A-Z][A-Z0-9_]+):/;

export function downloadErrorCode(download) {
  const explicit = typeof download?.error_code === "string" ? download.error_code.trim() : "";
  if (explicit) return explicit;
  const match = ERROR_CODE.exec(String(download?.error || ""));
  return match?.[1] || null;
}

export function isCivitaiPaidAccess(download) {
  if (downloadErrorCode(download) === "CIVITAI_PAID_ACCESS") return true;
  const text = `${download?.error || ""} ${download?.status_text || ""}`.toLowerCase();
  return /requires buzz|requires paid access|paid access required|paid licen[cs]e/.test(text);
}

export function isHfGatedAccess(download) {
  return downloadErrorCode(download) === "HF_GATED";
}

export function isAuthRequired(download) {
  return ["AUTH_REQUIRED", "HF_TOKEN_REQUIRED"].includes(downloadErrorCode(download));
}
