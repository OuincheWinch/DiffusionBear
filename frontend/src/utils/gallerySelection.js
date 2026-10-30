export function selectOnly(id) {
  return { ids: id ? [id] : [], anchor: id || null };
}

export function toggleSelection(ids, id) {
  if (!id) return { ids: Array.from(ids || []), anchor: null };
  const current = Array.from(ids || []);
  const next = current.includes(id) ? current.filter((item) => item !== id) : [...current, id];
  return { ids: next, anchor: id };
}

export function selectRange(visibleIds, anchorId, clickedId) {
  const ids = Array.isArray(visibleIds) ? visibleIds : [];
  const end = ids.indexOf(clickedId);
  if (end < 0) return { ids: [], anchor: anchorId || clickedId || null };

  const start = ids.indexOf(anchorId);
  if (start < 0) return { ids: [clickedId], anchor: clickedId };

  const lower = Math.min(start, end);
  const upper = Math.max(start, end);
  return { ids: ids.slice(lower, upper + 1), anchor: clickedId };
}
