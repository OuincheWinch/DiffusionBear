import { API_BASE, imageUrl } from "../api";

const generatedIdPattern = /^[a-f0-9]{16,64}$/i;

function referencePath(value) {
  if (typeof value === "string") return value;
  if (value && typeof value === "object") return value.path || value.file || value.id || "";
  return "";
}

export function normalizeRequest(request = {}) {
  const source = request && typeof request === "object" ? request : {};
  const format = source.output_format || source.format || "png";
  const refImages = Array.isArray(source.reference_images) && source.reference_images.length > 0
    ? source.reference_images
    : source.reference_image
      ? [source.reference_image]
      : Array.isArray(source.reference_images)
        ? source.reference_images
        : [];
  return {
    ...source,
    prompt: source.prompt ?? "",
    negative_prompt: source.negative_prompt ?? "",
    model: source.model || source.model_id || source.repo || "",
    width: source.width ?? 1024,
    height: source.height ?? 1024,
    steps: source.steps ?? 4,
    batch: source.batch ?? 1,
    sampler: source.sampler ?? "",
    cache_interval: source.cache_interval ?? 1,
    output_format: format,
    format,
    reference_images: refImages,
    reference_strength: source.reference_strength ?? source.image_strength ?? 0.6,
  };
}

export function referencePreview(value) {
  if (value && typeof value === "object") {
    if (value.preview && typeof value.preview === "string" && !value.preview.startsWith("blob:")) {
      return value.preview.startsWith("/api/") ? `${API_BASE}${value.preview}` : value.preview;
    }
    if (value.url && typeof value.url === "string") {
      return value.url.startsWith("/api/") ? `${API_BASE}${value.url}` : value.url;
    }
  }
  const path = referencePath(value);
  if (!path) return "";
  if (/^(https?:|blob:|data:)/i.test(path)) return path;
  if (path.startsWith("/api/")) return `${API_BASE}${path}`;
  if (path.includes("/uploads/")) {
    const filename = path.split(/[\\/]/).pop();
    return `${API_BASE}/api/uploads/${encodeURIComponent(filename)}`;
  }
  const clean = path.split(/[?#]/)[0];
  const filename = clean.split("/").pop() || clean;
  const id = filename.replace(/\.(png|jpe?g|webp|heic|heif)$/i, "");
  if (generatedIdPattern.test(id)) return imageUrl(id, true);
  if (id === filename && /^[a-z0-9_-]+$/i.test(id)) return imageUrl(id, true);
  // Default to uploads endpoint (backend automatically falls back to generated)
  return `${API_BASE}/api/uploads/${encodeURIComponent(filename)}`;
}

export function referenceItems(request) {
  const source = normalizeRequest(request);
  const values = source.reference_images;
  return values
    .map((value, index) => {
      const path = referencePath(value);
      if (!path) return null;
      const preview = referencePreview(value);
      const name = typeof value === "object" && value?.name
        ? value.name
        : (path.split(/[\\/]/).pop() || `Reference ${index + 1}`);
      return {
        id: `reference-${index}-${path}`,
        path,
        preview,
        name,
      };
    })
    .filter(Boolean);
}

export function resolveRequestModel(request, models = []) {
  const value = normalizeRequest(request).model;
  if (!value || !Array.isArray(models) || !models.length) return null;
  const target = String(value).trim().toLowerCase();
  // Exact or case-insensitive match on id, repo, model_id, or label
  const found = models.find((model) =>
    [model.id, model.repo, model.model_id, model.label].some(
      (candidate) => candidate && String(candidate).trim().toLowerCase() === target
    )
  );
  if (found) return found;

  // Legacy aliases / fuzzy matches for known model reorganizations
  if (target.includes("flux-3") || target.includes("flux.3")) {
    const flux3 = models.find((m) => m.id === "flux3-klein" || (m.repo && m.repo.includes("FLUX-3")));
    if (flux3) return flux3;
  }
  if (target.includes("qwen-image-2.1")) {
    const qwen = models.find((m) => m.id === "qwen-image-2.1" || m.id === "qwen-image-2.1-turbo");
    if (qwen) return qwen;
  }
  if (target === "pony" || target.includes("pony-diffusion")) {
    const pony = models.find((m) => m.id === "Pony" || m.id === "pony");
    if (pony) return pony;
  }
  if (target.includes("fibo")) {
    const fibo = models.find((m) => m.id === "fibo2-turbo" || (m.repo && m.repo.includes("FIBO")));
    if (fibo) return fibo;
  }
  return null;
}
