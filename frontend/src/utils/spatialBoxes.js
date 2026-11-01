// spatialBoxes.js — Spatial bounding box and ID extraction utilities

export const STOP_WORDS = new Set([
  "a", "an", "the", "this", "that", "these", "those",
  "with", "of", "in", "on", "at", "to", "from", "for",
  "and", "or", "but", "is", "are", "was", "were", "be",
  "holding", "wearing", "sitting", "standing", "lying",
  "some", "very", "extremely", "detailed", "clean", "smooth",
  "ultra", "macro", "close", "up", "shot", "render", "photo",
]);

export const RESIZE_HANDLES = ["nw", "n", "ne", "e", "se", "s", "sw", "w"];

export function extractNounId(desc, kind = "object", index = 1, lang = "En") {
  const idx = Math.max(1, parseInt(index, 10) || 1);
  if (kind === "text") {
    const langCode = (lang || "En").charAt(0).toUpperCase() + (lang || "En").slice(1).toLowerCase();
    return `${langCode}_Text_${idx}`;
  }
  if (!desc || typeof desc !== "string") {
    return `object_${idx}`;
  }
  const tokens = desc.toLowerCase().match(/[a-z0-9]+/g) || [];
  const meaningful = tokens.filter((t) => !STOP_WORDS.has(t) && !/^\d+$/.test(t));
  if (meaningful.length === 0) {
    return `object_${idx}`;
  }
  const headNoun = meaningful[meaningful.length - 1];
  return `${headNoun}_${idx}`;
}

export function clampBbox(ymin, xmin, ymax, xmax) {
  let y1 = Math.max(0, Math.min(1000, Math.round(ymin)));
  let x1 = Math.max(0, Math.min(1000, Math.round(xmin)));
  let y2 = Math.max(0, Math.min(1000, Math.round(ymax)));
  let x2 = Math.max(0, Math.min(1000, Math.round(xmax)));

  if (y1 > y2) [y1, y2] = [y2, y1];
  if (x1 > x2) [x1, x2] = [x2, x1];

  // Prevent zero-size boxes: ensure at least 8 units even at edges
  if (y2 - y1 < 8) {
    if (y2 >= 992) {
      y1 = Math.max(0, y2 - 8);
    } else {
      y2 = Math.min(1000, y1 + 8);
    }
  }
  if (x2 - x1 < 8) {
    if (x2 >= 992) {
      x1 = Math.max(0, x2 - 8);
    } else {
      x2 = Math.min(1000, x1 + 8);
    }
  }

  return [y1, x1, y2, x2];
}

export function parseBoxesFromPromptText(promptText, dims = {}) {
  if (!promptText || typeof promptText !== "string") return null;
  // Only trust caller-supplied pixel dimensions for the abs_centralized
  // conversion; zero/negative values fall back to per-axis estimation.
  const dimW = Number(dims.width) > 0 ? Number(dims.width) : null;
  const dimH = Number(dims.height) > 0 ? Number(dims.height) : null;

  // 1b. Try to parse MOSS layout JSON: `{"summary": ..., "items": [...]}`
  if (promptText.includes('"items"') && (promptText.includes('"bbox_format"') || promptText.includes('"bbox_coord_space"'))) {
    try {
      const parsed = JSON.parse(promptText.trim());
      if (parsed && Array.isArray(parsed.items) && parsed.items.length > 0) {
        return parsed.items.map((item, idx) => {
          const rawBbox = item.bbox || [0, 0, 0, 0];
          let y1 = rawBbox[0];
          let x1 = rawBbox[1];
          let y2 = rawBbox[2];
          let x2 = rawBbox[3];
          if (parsed.bbox_coord_space === "abs_centralized") {
            // Center-origin pixels on the real generation size. Prefer the
            // caller's actual width/height (the composer writes ±H/2 / ±W/2);
            // when unavailable, the extremes of the largest box reveal the
            // canvas size. Scaling by a fixed 512 assumed 1024×1024 and
            // crushed every 768×512 run into a stacked cluster top-left.
            const yHalf = dimH ? dimH / 2 : Math.max(
              1,
              ...parsed.items.map((it) => {
                const b = it.bbox || [0, 0, 0, 0];
                return Math.max(Math.abs(b[0]), Math.abs(b[2]));
              }),
            );
            const xHalf = dimW ? dimW / 2 : Math.max(
              1,
              ...parsed.items.map((it) => {
                const b = it.bbox || [0, 0, 0, 0];
                return Math.max(Math.abs(b[1]), Math.abs(b[3]));
              }),
            );
            y1 = Math.round(((y1 + yHalf) / (2 * yHalf)) * 1000);
            x1 = Math.round(((x1 + xHalf) / (2 * xHalf)) * 1000);
            y2 = Math.round(((y2 + yHalf) / (2 * yHalf)) * 1000);
            x2 = Math.round(((x2 + xHalf) / (2 * xHalf)) * 1000);
          }
          const [cy1, cx1, cy2, cx2] = clampBbox(y1, x1, y2, x2);
          const isText = item.label === "text";
          const desc = item.text || item.caption || "";
          return {
            uid: `moss_box_${idx}`,
            id: isText ? `En_Text_${idx + 1}` : `object_${idx + 1}`,
            bbox: [cy1, cx1, cy2, cx2],
            desc,
            kind: isText ? "text" : "object",
            action: "object",
            lang: "En",
          };
        });
      }
    } catch {
      // Incomplete JSON while typing, ignore
    }
  }

  // 1. Try to parse BFL JSON part: `[{ ... }]`
  const jsonMatch = promptText.match(/\[\s*\{[\s\S]*\}\s*\]/);
  if (jsonMatch) {
    try {
      const parsed = JSON.parse(jsonMatch[0]);
      if (Array.isArray(parsed) && parsed.length > 0) {
        return parsed.map((item, idx) => {
          const rawBbox = item.tgt_bbox || item.bbox || item.src_bbox || [250, 250, 750, 750];
          const [y1, x1, y2, x2] = clampBbox(rawBbox[0], rawBbox[1], rawBbox[2], rawBbox[3]);
          const isText = (item.id && /_text_/i.test(item.id)) || item.kind === "text";
          const action = item.action || (item.from === "ref_image_0" && !item.tgt_bbox ? "remove" : (item.src_bbox && item.tgt_bbox ? "move" : "add"));
          return {
            uid: item.uid || `box_prompt_${idx}_${item.id || idx + 1}`,
            id: item.id || (isText ? `En_Text_${idx + 1}` : `object_${idx + 1}`),
            bbox: [y1, x1, y2, x2],
            desc: item.desc || "",
            kind: isText ? "text" : "object",
            action,
            lang: item.lang || (item.id && item.id.split("_")[0]) || "En",
          };
        });
      }
    } catch {
      // Incomplete JSON while typing, ignore
    }
  }

  // 2. Try legacy bracket coordinates: `[ymin, xmin, ymax, xmax] label`
  const coordRegex = /\[\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*\]\s*([^,[\n]*)/g;
  const matches = [...promptText.matchAll(coordRegex)];
  if (matches.length > 0) {
    return matches.map((m, idx) => {
      const y1 = parseInt(m[1], 10);
      const x1 = parseInt(m[2], 10);
      const y2 = parseInt(m[3], 10);
      const x2 = parseInt(m[4], 10);
      const trailingDesc = (m[5] || "").trim();
      const isText = /"(.*?)"/.test(trailingDesc);
      const cleanDesc = isText ? trailingDesc.replace(/"/g, "").trim() : trailingDesc;
      const id = extractNounId(cleanDesc, isText ? "text" : "object", idx + 1, "En");
      return {
        uid: `box_coord_${idx}`,
        id: id || `object_${idx + 1}`,
        bbox: clampBbox(y1, x1, y2, x2),
        desc: cleanDesc,
        kind: isText ? "text" : "object",
        action: "object",
        lang: "En",
      };
    });
  }

  return null;
}

export function composeBoxesIntoPrompt(currentPrompt, boxes, hasRef = false) {
  if (!boxes || boxes.length === 0) {
    return (currentPrompt || "").replace(/\s*\[\{.*?\}\]\s*$/s, "").trim();
  }

  const prepared = boxes.map((b, i) => {
    const id = b.id || `object_${i + 1}`;
    const bbox = b.bbox || [0, 0, 1000, 1000];
    const desc = (b.desc || "").trim();
    const isRemove = b.action === "remove";
    const isMove = b.action === "move";
    return hasRef
      ? {
          id,
          from: (isRemove || isMove) ? "ref_image_0" : "target",
          src_bbox: (isRemove || isMove) ? bbox : null,
          tgt_bbox: isRemove ? null : bbox,
          desc,
        }
      : { id, bbox, desc };
  });

  const cleanPrompt = (currentPrompt || "").replace(/\s*\[\{.*?\}\]\s*$/s, "").trim();
  const isAutoTemplate = !cleanPrompt || /^(Modify <ref_image_0>:|A scene featuring)/i.test(cleanPrompt);
  let baseText = cleanPrompt;

  if (hasRef) {
    if (isAutoTemplate) {
      const actionsSummary = prepared.map((b) => `modify <${b.id}>`).join(", ");
      baseText = `Modify <ref_image_0>: ${actionsSummary}.`;
    } else {
      const validIds = new Set(["ref_image_0", ...prepared.map((b) => b.id)]);
      baseText = baseText.replace(/<([a-zA-Z0-9_]+)>/g, (m, tag) => (validIds.has(tag) ? m : ""));
      baseText = baseText.replace(/\s*,\s*,/g, ",").replace(/\s{2,}/g, " ").trim();
      if (!baseText.includes("<ref_image_0>")) {
        baseText = `Modify <ref_image_0>: ${baseText}`;
      }
      for (const b of prepared) {
        if (!baseText.includes(`<${b.id}>`)) {
          baseText = `${baseText.replace(/\.+$/, "")}, modify <${b.id}>.`;
        }
      }
    }
  } else {
    if (isAutoTemplate) {
      const idsSummary = prepared.map((b) => `<${b.id}>`).join(" and ");
      baseText = `A scene featuring ${idsSummary}.`;
    } else {
      const validIds = new Set(prepared.map((b) => b.id));
      baseText = baseText.replace(/<([a-zA-Z0-9_]+)>/g, (m, tag) => (validIds.has(tag) ? m : ""));
      baseText = baseText.replace(/\s*,\s*,/g, ",").replace(/\s{2,}/g, " ").trim();
      for (const b of prepared) {
        if (!baseText.includes(`<${b.id}>`)) {
          baseText = `${baseText.replace(/\.+$/, "")}, featuring <${b.id}>.`;
        }
      }
    }
  }

  const jsonStr = JSON.stringify(prepared);
  return `${baseText} ${jsonStr}`;
}


// Vivid, visually-distinct colors assigned per box (index-based so they stay
// stable while editing). `prompt-hex` aligned: the studio canvas uses these
// for the box outline/label, and the prompt editor highlights the matching
// descriptor text with the same value.
// Vivid, visually-distinct colors assigned per box (index-based so they stay
// stable while editing). 16 entries: the spatial canvas supports up to 16
// boxes. The studio canvas uses these for the box outline/label, and the
// prompt editor highlights the matching descriptor text with the same value.
export const BOX_COLORS = [
  "#ff3d7f", // neon pink
  "#35d0c8", // teal cyan
  "#ffd23f", // gold
  "#7c5cff", // violet
  "#4ade80", // mint green
  "#ff8a3d", // orange
  "#3da9ff", // sky blue
  "#ff5e5e", // coral red
  "#f472b6", // rose
  "#22d3ee", // aqua
  "#eab308", // amber - ok under dark UI
  "#34d399", // emerald
  "#60a5fa", // cornflower
  "#c084fc", // lavender purple
  "#fb7185", // salmon pink
  "#a3e635", // lime
];

export function boxColor(index) {
  return BOX_COLORS[((index % BOX_COLORS.length) + BOX_COLORS.length) % BOX_COLORS.length];
}
