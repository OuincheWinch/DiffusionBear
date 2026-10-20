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
    reference_images: Array.isArray(source.reference_images)
      ? source.reference_images
      : source.reference_image
        ? [source.reference_image]
        : [],
    reference_strength: source.reference_strength ?? source.image_strength ?? 0.6,
  };
}

export function referencePreview(value) {
  const path = referencePath(value);
  if (!path) return "";
  if (/^(https?:|blob:|data:)/i.test(path)) return path;
  if (path.startsWith("/api/")) return `${API_BASE}${path}`;
  if (path.startsWith("/")) {
    const name = path.split(/[\\/]/).pop()?.replace(/\.[^.]+$/, "") || "";
    return generatedIdPattern.test(name) ? imageUrl(name, true) : "";
  }
  const clean = path.split(/[?#]/)[0];
  const name = clean.split("/").pop() || clean;
  const id = name.replace(/\.(png|jpe?g|webp|heic|heif)$/i, "");
  if (generatedIdPattern.test(id)) return imageUrl(id, true);
  if (id === name && /^[a-z0-9_-]+$/i.test(id)) return imageUrl(id, true);
  return "";
}

export function referenceItems(request) {
  const source = normalizeRequest(request);
  const values = source.reference_images;
  return values
    .map((value, index) => {
      const path = referencePath(value);
      if (!path) return null;
      return {
        id: `reference-${index}-${path}`,
        path,
        preview: referencePreview(value),
        name: path.split(/[\\/]/).pop() || `Reference ${index + 1}`,
      };
    })
    .filter(Boolean);
}

export function resolveRequestModel(request, models) {
  const value = normalizeRequest(request).model;
  if (!value) return null;
  return models.find((model) =>
    [model.id, model.repo, model.model_id, model.label].some((candidate) => candidate === value)
  ) || null;
}
