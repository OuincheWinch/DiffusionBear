import math

import numpy as np


class AnimaPromptWeights:
    # ComfyUI's prompt weighting, as its Anima text encoder reads it: "(tag)" is weight 1.1, "(tag:1.3)" is 1.3,
    # nesting multiplies, and \( \) are literal parentheses. Each part is tokenized on its own; the Qwen3 encoder
    # sees every part at weight 1, and the adapter's output for each T5 token is scaled by its part's weight.
    # Port of parse_parentheses and token_weights in comfy/sd1_clip.py.

    MAX_WEIGHT = float(np.finfo(np.float32).max)

    @staticmethod
    def parse(prompt: str) -> list[tuple[str, float]]:
        escaped = prompt.replace("\\)", "\0\1").replace("\\(", "\0\2")
        return [
            (part.replace("\0\1", ")").replace("\0\2", "("), weight)
            for part, weight in AnimaPromptWeights._weights(escaped, 1.0)
        ]

    @staticmethod
    def unescape(text: str) -> str:
        # \( and \) as parse leaves them, for text read without weights
        return text.replace("\\(", "(").replace("\\)", ")")

    @staticmethod
    def is_plain(parts: list[tuple[str, float]]) -> bool:
        # No weights and one part (an empty prompt has none): the prompt is tokenized whole, as before
        return len(parts) <= 1 and all(weight == 1.0 for _, weight in parts)

    @staticmethod
    def _weights(text: str, current_weight: float) -> list[tuple[str, float]]:
        out = []
        for part in AnimaPromptWeights._parentheses(text):
            if len(part) >= 2 and part[0] == "(" and part[-1] == ")":
                inner = part[1:-1]
                weight = current_weight * 1.1
                colon = inner.rfind(":")
                if colon > 0:
                    try:
                        weight = float(inner[colon + 1 :])
                        inner = inner[:colon]
                    except ValueError:
                        pass
                # The weights go to the encoder as a float32 array. ComfyUI would scale the context by a weight
                # that is not finite and the image comes out black. Nested groups multiply, so check the product.
                if not math.isfinite(weight) or abs(weight) > AnimaPromptWeights.MAX_WEIGHT:
                    raise ValueError(f"The prompt weight in ({part[1:-1]}) is not a finite number in float32 range.")
                out += AnimaPromptWeights._weights(inner, weight)
            else:
                out.append((part, current_weight))
        return out

    @staticmethod
    def _parentheses(text: str) -> list[str]:
        result = []
        current = ""
        depth = 0
        for char in text:
            if char == "(":
                if depth == 0:
                    if current:
                        result.append(current)
                    current = "("
                else:
                    current += char
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    result.append(current + ")")
                    current = ""
                else:
                    current += char
            else:
                current += char
        if current:
            result.append(current)
        return result
