"""
BFL FLUX.3-klein Spatial Box Composer & Coordinate Engine.
Adheres strictly to Black Forest Labs' official FLUX.3-klein protocol:
- [ymin, xmin, ymax, xmax] on a 0..1000 integer grid (Vertical Y first, Horizontal X second).
- Sorted left-to-right (xmin), then top-to-bottom (ymin).
- Noun IDs: <noun>_<index> (e.g. hat_1, clock_2).
- Verbatim Text IDs: <Lang>_Text_<index> (e.g. En_Text_1, Fr_Text_2).
- T2I and Image Editing (ref_image_0) prompt synthesis and serialization.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

_STOP_WORDS = {
    "a", "an", "the", "this", "that", "these", "those",
    "with", "of", "in", "on", "at", "to", "from", "for",
    "and", "or", "but", "is", "are", "was", "were", "be",
    "holding", "wearing", "sitting", "standing", "lying",
    "some", "very", "extremely", "detailed", "clean", "smooth",
    "ultra", "macro", "close", "up", "shot", "render", "photo",
}


def validate_and_clamp_bbox(bbox: Any) -> List[int]:
    """
    Validate, clamp to 0..1000, and ensure ymin <= ymax and xmin <= xmax.
    Input must be a sequence of 4 numbers [ymin, xmin, ymax, xmax].
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return [0, 0, 1000, 1000]

    try:
        coords = [int(round(float(v))) for v in bbox]
    except (ValueError, TypeError):
        return [0, 0, 1000, 1000]

    # Clamp to 0..1000
    ymin = max(0, min(1000, coords[0]))
    xmin = max(0, min(1000, coords[1]))
    ymax = max(0, min(1000, coords[2]))
    xmax = max(0, min(1000, coords[3]))

    # Ensure min <= max
    if ymin > ymax:
        ymin, ymax = ymax, ymin
    if xmin > xmax:
        xmin, xmax = xmax, xmin

    return [ymin, xmin, ymax, xmax]


def extract_id_from_desc(
    desc: str,
    kind: str = "object",
    index: int = 1,
    lang: str = "En"
) -> str:
    """
    Generate clean BFL identifier:
    - Text kind: <Lang>_Text_<index> (e.g. En_Text_1)
    - Object kind: Extracts the head noun from description (e.g. "cowboy hat" -> hat_1)
    """
    idx = max(1, int(index))
    if kind == "text":
        lang_code = (lang or "En").capitalize()
        return f"{lang_code}_Text_{idx}"

    if not desc or not isinstance(desc, str):
        return f"object_{idx}"

    # Extract words
    tokens = re.findall(r"[A-Za-z0-9]+", desc.lower())
    meaningful = [t for t in tokens if t not in _STOP_WORDS and not t.isdigit()]

    if not meaningful:
        return f"object_{idx}"

    # In English noun phrases, the head noun is usually the last word
    head_noun = meaningful[-1]
    return f"{head_noun}_{idx}"


def compose_flux3_bbox_prompt(
    global_prompt: str,
    boxes: List[Dict[str, Any]],
    has_reference: bool = False
) -> str:
    """
    Compose final BFL single-line prompt with structured bounding box payload.
    Supports both:
    1. T2I Format (has_reference=False):
       <caption_with_<id>> [{"id": "...", "bbox": [...], "desc": "..."}]
    2. Image Editing Format (has_reference=True):
       <instruction_with_<id>_and_<ref_image_0>> [{"id": "...", "from": "...", "src_bbox": [...], "tgt_bbox": [...], "desc": "..."}]
    """
    if not boxes:
        return global_prompt or ""

    clean_prompt = (global_prompt or "").strip()

    if not has_reference:
        # T2I Mode
        prepared_boxes = []
        for i, b in enumerate(boxes, start=1):
            box_id = b.get("id") or extract_id_from_desc(
                b.get("desc", ""),
                kind=b.get("kind", "object"),
                index=i,
                lang=b.get("lang", "En")
            )
            raw_bbox = b.get("bbox") or b.get("tgt_bbox") or [0, 0, 1000, 1000]
            bbox = validate_and_clamp_bbox(raw_bbox)
            desc = (b.get("desc") or "").strip()
            prepared_boxes.append({
                "id": box_id,
                "bbox": bbox,
                "desc": desc,
            })

        # Sort left-to-right (xmin), then top-to-bottom (ymin)
        prepared_boxes.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))

        # Synthesize or augment global prompt
        all_ids = [b["id"] for b in prepared_boxes]
        if not clean_prompt:
            id_list = " and ".join(f"<{b_id}>" for b_id in all_ids)
            clean_prompt = f"A high quality image featuring {id_list}."
        else:
            missing_ids = [b_id for b_id in all_ids if f"<{b_id}>" not in clean_prompt]
            if missing_ids:
                suffix = ", featuring " + ", ".join(f"<{b_id}>" for b_id in missing_ids)
                clean_prompt = clean_prompt.rstrip(". ") + suffix + "."

        payload_json = json.dumps(prepared_boxes, ensure_ascii=False)
        return f"{clean_prompt} {payload_json}"

    else:
        # Image Editing Mode (with ref_image_0)
        prepared_boxes = []
        for i, b in enumerate(boxes, start=1):
            action = b.get("action", "add")
            box_id = b.get("id") or extract_id_from_desc(
                b.get("desc", ""),
                kind=b.get("kind", "object"),
                index=i,
                lang=b.get("lang", "En")
            )
            desc = (b.get("desc") or "").strip()

            if action in ("remove", "move"):
                from_source = "ref_image_0"
            else:
                from_source = "target"

            # Resolve coordinates defensively regardless of whether frontend sent
            # generic `bbox` or explicit `src_bbox` / `tgt_bbox`.
            if action == "remove":
                raw_src = b.get("src_bbox") or b.get("bbox")
                raw_tgt = b.get("tgt_bbox")
            elif action == "move":
                raw_src = b.get("src_bbox")
                raw_tgt = b.get("tgt_bbox") or b.get("bbox")
            else:
                # add / replace / object / text / default
                raw_src = b.get("src_bbox")
                raw_tgt = b.get("tgt_bbox") or b.get("bbox")

            src_bbox = validate_and_clamp_bbox(raw_src) if raw_src else None
            tgt_bbox = validate_and_clamp_bbox(raw_tgt) if raw_tgt else None

            prepared_boxes.append({
                "id": box_id,
                "from": from_source,
                "src_bbox": src_bbox,
                "tgt_bbox": tgt_bbox,
                "desc": desc,
            })

        # Sort by primary active coordinate
        def _sort_key(item):
            coord = item["tgt_bbox"] or item["src_bbox"] or [0, 0, 1000, 1000]
            return (coord[1], coord[0])

        prepared_boxes.sort(key=_sort_key)

        all_ids = [b["id"] for b in prepared_boxes]
        if not clean_prompt:
            actions_summary = ", ".join(f"modify <{b['id']}>" for b in prepared_boxes)
            clean_prompt = f"Modify <ref_image_0>: {actions_summary}."
        else:
            if "<ref_image_0>" not in clean_prompt:
                clean_prompt = f"Modify <ref_image_0>: {clean_prompt}"
            missing_ids = [b_id for b_id in all_ids if f"<{b_id}>" not in clean_prompt]
            if missing_ids:
                clean_prompt = clean_prompt.rstrip(". ") + f" with {', '.join(f'<{b_id}>' for b_id in missing_ids)}."

        payload_json = json.dumps(prepared_boxes, ensure_ascii=False)
        return f"{clean_prompt} {payload_json}"


def serialize_moss_boxes(
    boxes: List[Dict[str, Any]],
    width: int = 1024,
    height: int = 1024,
    summary_caption: str = "",
    meta: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Serialize spatial boxes into official OpenMOSS MOSS-Image structured JSON layout.
    Schema:
    {
      "summary": {
        "meta": meta or ["scene"],
        "canvas": "single",
        "caption": summary_caption
      },
      "items": [
        {
          "label": "text" | "object",
          "caption": "...",
          "text": "...", # optional, only when label is text
          "bbox": [ymin, xmin, ymax, xmax] # abs_centralized: [-H/2..H/2, -W/2..W/2]
        }
      ],
      "bbox_coord_space": "abs_centralized",
      "bbox_format": "yxyx"
    }
    """
    w = float(width or 1024)
    h = float(height or 1024)
    items: List[Dict[str, Any]] = []

    for i, b in enumerate(boxes, start=1):
        raw_bbox = b.get("bbox") or b.get("tgt_bbox") or [0, 0, 1000, 1000]
        y1, x1, y2, x2 = validate_and_clamp_bbox(raw_bbox)

        # Transform 0..1000 normalized grid to abs_centralized coordinates
        # Canvas center is (0, 0). Coordinates range from -H/2..H/2 and -W/2..W/2.
        ymin_c = int(round((y1 / 1000.0 - 0.5) * h))
        xmin_c = int(round((x1 / 1000.0 - 0.5) * w))
        ymax_c = int(round((y2 / 1000.0 - 0.5) * h))
        xmax_c = int(round((x2 / 1000.0 - 0.5) * w))

        desc = (b.get("desc") or "").strip()
        kind = b.get("kind") or "object"
        is_text = kind == "text" or (desc.startswith('"') and desc.endswith('"'))

        if is_text:
            text_val = desc.strip('"').strip()
            item_entry: Dict[str, Any] = {
                "label": "text",
                "caption": desc or text_val,
                "text": text_val,
                "bbox": [ymin_c, xmin_c, ymax_c, xmax_c],
            }
        else:
            item_entry = {
                "label": "object",
                "caption": desc or f"object_{i}",
                "bbox": [ymin_c, xmin_c, ymax_c, xmax_c],
            }

        items.append(item_entry)

    # Sort items left-to-right (xmin_c), then top-to-bottom (ymin_c)
    items.sort(key=lambda it: (it["bbox"][1], it["bbox"][0]))

    return {
        "summary": {
            "meta": meta or ["scene"],
            "canvas": "single",
            "caption": summary_caption or "",
        },
        "items": items,
        "bbox_coord_space": "abs_centralized",
        "bbox_format": "yxyx",
    }


def compose_moss_bbox_prompt(
    global_prompt: str,
    boxes: List[Dict[str, Any]],
    width: int = 1024,
    height: int = 1024,
) -> str:
    """
    Compose final MOSS-Image layout JSON string from prompt and spatial boxes.
    If no boxes, returns global_prompt unchanged.
    """
    if not boxes:
        return global_prompt or ""

    clean_caption = (global_prompt or "").strip()
    layout = serialize_moss_boxes(
        boxes=boxes,
        width=width,
        height=height,
        summary_caption=clean_caption,
    )
    return json.dumps(layout, ensure_ascii=False)


def parse_moss_boxes(
    moss_json_or_str: Any,
    width: int = 1024,
    height: int = 1024,
) -> List[Dict[str, Any]]:
    """
    Parse MOSS-Image layout JSON back into DiffusionBear 0..1000 grid boxes.
    """
    if isinstance(moss_json_or_str, str):
        try:
            data = json.loads(moss_json_or_str)
        except Exception:
            return []
    elif isinstance(moss_json_or_str, dict):
        data = moss_json_or_str
    else:
        return []

    items = data.get("items")
    if not isinstance(items, list):
        return []

    w = float(width or 1024)
    h = float(height or 1024)
    coord_space = data.get("bbox_coord_space", "abs_centralized")
    result: List[Dict[str, Any]] = []

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        raw_bbox = item.get("bbox") or [0, 0, 0, 0]
        if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
            continue

        if coord_space == "abs_centralized":
            y1 = int(round(((raw_bbox[0] + h / 2.0) / h) * 1000.0))
            x1 = int(round(((raw_bbox[1] + w / 2.0) / w) * 1000.0))
            y2 = int(round(((raw_bbox[2] + h / 2.0) / h) * 1000.0))
            x2 = int(round(((raw_bbox[3] + w / 2.0) / w) * 1000.0))
        else:
            y1, x1, y2, x2 = raw_bbox

        clamped = validate_and_clamp_bbox([y1, x1, y2, x2])
        label = item.get("label", "object")
        desc = item.get("text") or item.get("caption") or ""
        box_id = f"{label}_{idx + 1}"

        result.append({
            "uid": f"moss_box_{idx}_{box_id}",
            "id": box_id,
            "bbox": clamped,
            "desc": desc,
            "kind": "text" if label == "text" else "object",
            "action": "object",
        })

    return result

