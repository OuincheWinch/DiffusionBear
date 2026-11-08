export function hasReferenceInput(item) {
  if (!item) return false;
  const refs = item.reference_images;
  if (Array.isArray(refs)) {
    if (refs.some((value) => String(value || "").trim())) return true;
  } else if (String(refs || "").trim()) {
    return true;
  }
  return ["reference_image", "mask_file", "mask_image", "mask_path", "filled_from"].some(
    (key) => String(item[key] || "").trim()
  );
}
