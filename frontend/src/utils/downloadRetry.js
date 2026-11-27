export function retryRequestFor(download) {
  const direct = typeof download?.retry_input === "string" ? download.retry_input.trim() : "";
  const version = String(download?.version_id ?? "").trim();
  const input = direct || (/^\d+$/.test(version) ? version : "");
  if (!input) return null;
  const name = typeof download?.model_name === "string" ? download.model_name.trim() : "";
  return {
    url_or_id: input,
    ...(name ? { name } : {}),
  };
}
