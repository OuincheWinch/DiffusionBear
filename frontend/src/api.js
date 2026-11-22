// In the packaged app the backend serves this SPA from the SAME port it serves the
// API on, so a relative base is both correct and port-agnostic -- which matters
// because the launcher may move the backend to 9001, 10049 and so on when 8001 is
// held by something else. An absolute base baked the port into the bundle and would
// have talked to the wrong process, or nothing at all.
//
// Only the vite dev server is a different origin (5173), and there the backend is
// on 8001. VITE_API_BASE still wins, for anyone who needs to point it elsewhere.
const DEV_API_BASE = "http://localhost:8001";
export const API_BASE_STORAGE_KEY = "DIFFUSIONBEAR_API_BASE";

export function normalizeApiBase(value) {
  const raw = String(value ?? "").trim();
  if (!raw) return "";
  try {
    const parsed = new URL(raw);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") return "";
    return raw.replace(/\/+$/, "");
  } catch {
    return "";
  }
}

function storedApiBase() {
  try {
    return normalizeApiBase(globalThis.localStorage?.getItem(API_BASE_STORAGE_KEY));
  } catch {
    return "";
  }
}

export const API_BASE = (
  storedApiBase()
  || normalizeApiBase(import.meta.env?.VITE_API_BASE)
  || (import.meta.env?.DEV ? DEV_API_BASE : "")
).replace(/\/+$/, "");
// A build-time token is public by definition: it ships inside /assets/*.js,
// which the SPA serves without auth. Keep it dev-only; production uses the
// runtime key field stored in localStorage.
export const API_TOKEN = import.meta.env?.DEV
  ? (import.meta.env?.VITE_API_TOKEN || "")
  : "";
export const API_TOKEN_HEADER = import.meta.env?.VITE_API_TOKEN_HEADER || "Authorization";
export const API_TOKEN_SCHEME = import.meta.env?.VITE_API_TOKEN_SCHEME ?? "Bearer";
// PRO deployments can set this key at runtime from the browser console or a
// future settings panel without rebuilding the bundle. The build-time token
// remains the fallback for packaged installations.
export const API_TOKEN_STORAGE_KEY = "DIFFUSIONBEAR_API_KEY";

export function isClientMode() {
  const base = storedApiBase();
  if (!base) return false;
  try {
    const host = new URL(base).hostname.toLowerCase();
    return !["localhost", "127.0.0.1", "::1", "[::1]"].includes(host);
  } catch {
    return false;
  }
}

export function runtimeApiToken() {
  try {
    const stored = globalThis.localStorage?.getItem(API_TOKEN_STORAGE_KEY);
    if (stored && stored.trim()) return stored.trim();
  } catch {
    // Private mode or a non-browser test environment: fall through to the build.
  }
  return API_TOKEN;
}

export function authHeaders(input = {}) {
  const headers = typeof Headers !== "undefined" && input instanceof Headers
    ? new Headers(input)
    : { ...input };
  const token = runtimeApiToken();
  if (!token) return headers;
  // Object.keys() on a Headers instance returns [] -- headers are not own
  // enumerable properties -- so the check below never fires for one and a
  // caller-passed Authorization is silently overwritten. Headers.has() is
  // case-insensitive, which is also what the header spec requires.
  if (typeof headers.has === "function") {
    if (headers.has(API_TOKEN_HEADER)) return headers;
  } else {
    const existing = Object.keys(headers).find((key) => key.toLowerCase() === API_TOKEN_HEADER.toLowerCase());
    if (existing) return headers;
  }
  const value = API_TOKEN_SCHEME
    ? `${API_TOKEN_SCHEME} ${token}`.trim()
    : token;
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
    let detail;
    try {
      const data = await res.json();
      detail = data?.detail;
      if (typeof detail === "string") msg = detail;
      else if (detail) msg = JSON.stringify(detail);
    } catch {
      // Not an error in itself: an HTML error page or an empty body has no `detail` to
      // extract, and the status line is the best message available. Recorded so the
      // fall-through below is obviously deliberate rather than an ignored failure.
      msg = msg || `HTTP ${res.status}`;
    }
    const error = new Error(msg);
    error.status = res.status;
    // Structured details (for example steps_requires_ack) must survive as an
    // object so the form can react to the code instead of parsing a message.
    error.detail = detail;
    throw error;
  }
  if (res.status === 204) return null;
  return res.json();
}

export function withMediaToken(url) {
  if (!url) return url;
  const token = runtimeApiToken();
  if (!token) return url;
  const sep = url.includes("?") ? "&" : "?";
  return `${url}${sep}token=${encodeURIComponent(token)}`;
}

export function imageUrl(id, thumb = false, transparent = false) {
  const params = new URLSearchParams();
  if (thumb) params.set("thumb", "true");
  if (thumb && transparent) params.set("alpha", "1");
  const token = runtimeApiToken();
  // <img> cannot send Authorization. The backend accepts ?token= only on GET
  // media routes; every JSON or mutating route still requires the header.
  if (token) params.set("token", token);
  const query = params.toString();
  return `${API_BASE}/api/images/${encodeURIComponent(id)}/file${query ? `?${query}` : ""}`;
}

export async function fetchImageBlob(id) {
  const url = imageUrl(id);
  let res;
  try {
    res = await fetch(url, { headers: authHeaders() });
  } catch (err) {
    // A bare fetch() rejection is a bare TypeError whose message is the fixed string
    // "Failed to fetch" -- identical whether the backend is down, the port moved, the
    // request was blocked as cross-origin, or the connection dropped mid-body. Showing
    // that verbatim tells the user nothing they can act on, so name the URL instead.
    // `transport` marks it as "never got a response", which is what the download
    // fallback below keys on; an HTTP error DOES have a response and is not retried.
    const error = new Error(
      `Could not reach the backend for ${url} (${err?.message || "network error"}).`
    );
    error.transport = true;
    throw error;
  }
  if (!res.ok) {
    const error = new Error(`Failed to fetch image: ${res.statusText || res.status}`);
    error.transport = false;
    error.status = res.status;
    throw error;
  }
  return res.blob();
}

function triggerLinkDownload(url, filename) {
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
}

export async function downloadImage(id, filename) {
  const name = filename || `${id}.png`;
  try {
    const blob = await fetchImageBlob(id);
    const objectUrl = URL.createObjectURL(blob);
    triggerLinkDownload(objectUrl, name);
    window.setTimeout(() => URL.revokeObjectURL(objectUrl), 10000);
    return;
  } catch (err) {
    // A link navigation is not subject to the fetch/CORS rules that made the request
    // above fail, and it is the path Gallery already uses. So when the blob read never
    // reached the server, try the navigation before giving up -- but only then: an HTTP
    // error means the server DID answer, and retrying it the same way just hides a
    // real 404 behind a silent empty file.
    if (!err?.transport) throw err;
    triggerLinkDownload(imageUrl(id), name);
  }
}

/**
 * Can this LoRA be applied to this model?
 *
 * Returns `{compatible, unknown, message}`. `compatible` is null when the answer could
 * not be determined (weights not downloaded, model not mflux) -- which is NOT the same
 * as true, and must not be rendered as "fine". A caller that treats unknown as a pass
 * has reintroduced the bug this endpoint exists to fix.
 */
export async function loraCompatibility(name, modelId) {
  return api(`/api/loras/${encodeURIComponent(name)}/compatibility?model=${encodeURIComponent(modelId)}`);
}

export async function pickPath({ kind = "folder", title, extensions } = {}) {
  return api("/api/system/pick-path", {
    method: "POST",
    body: JSON.stringify({ kind, title, extensions }),
  });
}
