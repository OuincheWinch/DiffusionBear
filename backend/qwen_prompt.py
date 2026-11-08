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
