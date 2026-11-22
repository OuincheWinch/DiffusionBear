"""
FLUX.3 Spatial Bounding Box Parser & Validator.
Coordinates are represented on a normalized 0-1000 integer grid: [ymin, xmin, ymax, xmax].
"""
from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple

BOX_PATTERN = re.compile(
    r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]"
)


def validate_bounding_box(box: Any) -> bool:
    """
    Validate a single bounding box formatted as [ymin, xmin, ymax, xmax].
    All coordinates must be integers in the range [0, 1000] with ymin <= ymax and xmin <= xmax.
    """
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in box):
        return False
    ymin, xmin, ymax, xmax = box
    if not (0 <= ymin <= 1000 and 0 <= xmin <= 1000 and 0 <= ymax <= 1000 and 0 <= xmax <= 1000):
        return False
    if ymin > ymax or xmin > xmax:
        return False
    return True


def parse_bounding_boxes(prompt: Optional[str]) -> Tuple[List[List[int]], str]:
    """
    Extract valid bounding boxes from prompt text.
    Returns (boxes, prompt).
    """
    if not prompt or not isinstance(prompt, str):
        return [], ""
    matches = BOX_PATTERN.findall(prompt)
    boxes: List[List[int]] = []
    for match in matches:
        box = [int(v) for v in match]
        if validate_bounding_box(box):
            boxes.append(box)
    return boxes, prompt


def format_bounding_box_prompt(ymin: int, xmin: int, ymax: int, xmax: int, label: str) -> str:
    """
    Format bounding box coordinates and descriptive label into prompt syntax:
    "[ymin, xmin, ymax, xmax] label".
    """
    return f"[{ymin}, {xmin}, {ymax}, {xmax}] {label.strip()}"
