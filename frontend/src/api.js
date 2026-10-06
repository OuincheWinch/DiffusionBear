// In the packaged app the backend serves this SPA from the SAME port it serves the
// API on, so a relative base is both correct and port-agnostic -- which matters
// because the launcher may move the backend to 9001, 10049 and so on when 8001 is
// held by something else. An absolute base baked the port into the bundle and would
// have talked to the wrong process, or nothing at all.
//
// Only the vite dev server is a different origin (5173), and there the backend is
// on 8001. VITE_API_BASE still wins, for anyone who needs to point it elsewhere.
const DEV_API_BASE = "http://localhost:8001";
export const API_BASE = (
  import.meta.env?.VITE_API_BASE
  ?? (import.meta.env?.DEV ? DEV_API_BASE : "")
).replace(/\/+$/, "");
export const API_TOKEN = import.meta.env?.VITE_API_TOKEN || "";
export const API_TOKEN_HEADER = import.meta.env?.VITE_API_TOKEN_HEADER || "Authorization";
export const API_TOKEN_SCHEME = import.meta.env?.VITE_API_TOKEN_SCHEME ?? "Bearer";

export function authHeaders(input = {}) {
  const headers = typeof Headers !== "undefined" && input instanceof Headers
    ? new Headers(input)
    : { ...input };
  if (!API_TOKEN) return headers;
  const existing = Object.keys(headers).find((key) => key.toLowerCase() === API_TOKEN_HEADER.toLowerCase());
  if (existing) return headers;
  const value = API_TOKEN_SCHEME
    ? `${API_TOKEN_SCHEME} ${API_TOKEN}`.trim()
    : API_TOKEN;
  if (typeof headers.set === "function") {
    headers.set(API_TOKEN_HEADER, value);
    return headers;
  }
  return { ...headers, [API_TOKEN_HEADER]: value };
}

export async function api(path, options = {}) {
  const isFormData = typeof FormData !== "undefined" && options.body instanceof FormData;
  const headers = authHeaders(options.headers || {});
  if (!isFormData && options.body && typeof headers.set === "function" && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  } else if (!isFormData && options.body && typeof headers.set !== "function" && !Object.keys(headers).some((key) => key.toLowerCase() === "content-type")) {
    headers["Content-Type"] = "application/json";
  }
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
  });
  if (!res.ok) {
    let msg = res.statusText || `HTTP ${res.status}`;
    try {
      const data = await res.json();
      if (typeof data?.detail === "string") msg = data.detail;
      else if (data?.detail) msg = JSON.stringify(data.detail);
    } catch {}
    const error = new Error(msg);
    error.status = res.status;
    throw error;
  }
  if (res.status === 204) return null;
  return res.json();
}

export function imageUrl(id, thumb = false) {
  return `${API_BASE}/api/images/${encodeURIComponent(id)}/file${thumb ? "?thumb=true" : ""}`;
}

export async function fetchImageBlob(id) {
  const res = await fetch(imageUrl(id), { headers: authHeaders() });
  if (!res.ok) throw new Error(`Failed to fetch image: ${res.statusText || res.status}`);
  return res.blob();
}

export async function downloadImage(id, filename) {
  const blob = await fetchImageBlob(id);
  const objectUrl = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = objectUrl;
  link.download = filename || `${id}.png`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(objectUrl), 0);
}
