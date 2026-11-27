export function isUnlinkAction(m) {
  if (m?.remove_action) return m.remove_action === "unlink";
  return Boolean(m?.local_path || m?.local_path_configured || m?.is_builtin === false);
}

export function removeModelEndpoint(m) {
  const id = encodeURIComponent(m?.id || "");
  if (m?.is_builtin === false) {
    return `/api/custom-models/${id}`;
  }
  return `/api/models/${id}`;
}
