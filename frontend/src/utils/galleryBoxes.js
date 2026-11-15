import { clampBbox } from "./spatialBoxes.js";

function validBox(raw) {
  if (!Array.isArray(raw) || raw.length !== 4) return null;
  if (!raw.every((value) => Number.isFinite(Number(value)))) return null;
  return clampBbox(raw[0], raw[1], raw[2], raw[3]);
}

/**
 * Normalise the two sidecar shapes into one read-only overlay list.
 * `spatial_boxes` carries the user's local prompt in `desc`; legacy
 * `bounding_boxes` is geometry only and falls back to a numbered label.
 */
export function galleryBoxes(item) {
  if (!item || typeof item !== "object") return [];
  const spatial = Array.isArray(item.spatial_boxes) ? item.spatial_boxes : [];
  const fromSpatial = spatial
    .map((box, index) => {
      if (!box || typeof box !== "object") return null;
      const bbox = validBox(box.tgt_bbox || box.bbox || box.src_bbox);
      if (!bbox) return null;
      return {
        key: String(box.id || `box_${index + 1}`),
        id: String(box.id || `box_${index + 1}`),
        desc: typeof box.desc === "string" ? box.desc.trim() : "",
        kind: box.kind === "text" ? "text" : "object",
        bbox,
      };
    })
    .filter(Boolean);
  if (fromSpatial.length) return fromSpatial;

  const legacy = Array.isArray(item.bounding_boxes) ? item.bounding_boxes : [];
  return legacy
    .map((raw, index) => {
      const bbox = validBox(raw);
      if (!bbox) return null;
      return {
        key: `bounding_${index + 1}`,
        id: `box_${index + 1}`,
        desc: "",
        kind: "object",
        bbox,
      };
    })
    .filter(Boolean);
}

export function hasGalleryBoxes(item) {
  return galleryBoxes(item).length > 0;
}
