"""Shared Qwen prompt-tag normalisation.

`generator` and the `qwen_engine` subprocess both receive the same prompt, so the
regex lives here once instead of being copied into both modules.
"""

import re


def normalize_qwen_prompt(prompt: str) -> str:
    """Normalize user reference image tags (e.g. 'Image 1') to Qwen's 'Picture N' format.

    Protects model name references such as 'Qwen Image 2.1' or decimal versions.
    """
    if not prompt or not isinstance(prompt, str):
        return prompt or ""
    pattern = r"(?<!Qwen\s)(?<!Qwen-)\bImage\s+([1-9]\d*)(?!\.\d)\b"
    return re.sub(pattern, r"Picture \1", prompt, flags=re.IGNORECASE)


def wrap_qwen_transparent_prompt(prompt: str) -> str:
    """Wrap a prompt with Qwen-Image 2.1's native RGBA transparency conditioning.

    Leaves prompts that already specify an RGBA/alpha transparency wrapper intact.
    """
    text = (prompt or "").strip()
    lower = text.lower()
    if "rgba image with transparency" in lower or "alpha channel" in lower:
        return text
    if not text:
        return "This is an RGBA image with transparency. The image has alpha channel and the background is transparent."
    core = text.rstrip(".")
    return (
        f"This is an RGBA image with transparency. {core}. "
        "The image has alpha channel and the background is transparent."
    )
