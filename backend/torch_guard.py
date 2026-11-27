"""Keep torch out of the standard MLX inference process."""
from __future__ import annotations

import os
import sys

_TORCH_MODULE_NAMES = (
    "torch",
    "torch.nn",
    "torch.utils",
    "safetensors.torch",
)


def install() -> None:
    """Purge any stale torch module entries without planting a fake stub.

    Because torch is absent from the runtime environment, leaving sys.modules
    clean lets ``import torch`` fail naturally with ``ModuleNotFoundError`` and
    avoids ``transformers`` printing ``found 0.0.0+diffusionbear-stub``.
    """
    if os.environ.get("DIFFUSIONBEAR_ALLOW_TORCH") == "1":
        return
    for name in _TORCH_MODULE_NAMES:
        sys.modules.pop(name, None)


def release() -> None:
    """Clear any torch module entries added during an isolated import pass."""
    for name in _TORCH_MODULE_NAMES:
        sys.modules.pop(name, None)
