#!/usr/bin/env python3
"""Text-to-image with Bria's FIBO 2.0 Edit Turbo on Apple silicon, with Apple's MLX: no PyTorch needed.

The same model and the same steps as fibo2_edit_turbo_t2i.py (diffusers), run in MLX: 4 steps, no classifier-free
guidance. The weights are read as they are from Bria's repository (https://huggingface.co/briaai/Fibo-2.0-edit-turbo,
downloaded on the first run); nothing is converted or stored here. This one file holds the code around them:

  - the FIBO 2.0 transformer, ported module for module from diffusers' BriaFibo2Transformer2DModel (Apache-2.0);
  - the text encoder, Qwen3-VL-4B-Instruct, through mlx-vlm's Qwen3-VL;
  - the decoder of the FLUX.2 VAE that FIBO 2.0 uses: its layers, in code (the weights come from the model's vae/);
  - the steps of diffusers' BriaFibo2Pipeline: the prompt forms, the images of a caption's text, the schedule.

By default the text encoder first writes the prompt into a VGL caption, given bria-vgl's open "generate" instructions
(https://github.com/Bria-AI/vgl, MIT, fetched once and cached), then the caption is rendered and saved next to the image.

Examples:
    python fibo2_edit_turbo_t2i_mlx.py "a red vintage bicycle leaning against a whitewashed wall on a narrow street"
    python fibo2_edit_turbo_t2i_mlx.py "an origami whale above a paper city" --width 1424 --height 800 --seed 42
    python fibo2_edit_turbo_t2i_mlx.py --json assets/lake-pier.json --seed 4242     # a finished caption: no writing step
    python fibo2_edit_turbo_t2i_mlx.py --plain "A lighthouse on a cliff at dusk."      # the prompt as typed, no caption
    python fibo2_edit_turbo_t2i_mlx.py --sample 2 --seed 7 "a lighthouse on a cliff at dusk"   # another take on the caption
    python fibo2_edit_turbo_t2i_mlx.py --prompts examples/prompts.json --out-dir out --seed 4242   # a batch

Requirements (requirements-mlx.txt): an Apple silicon Mac, mlx, mlx-vlm, transformers, huggingface_hub, Pillow 10.1 or
newer and numpy. PyTorch is optional: when it is installed, it draws the starting noise (nothing else), so a seed gives
the same image as fibo2_edit_turbo_t2i.py. The model is gated: accept its licence at
https://huggingface.co/briaai/Fibo-2.0-edit-turbo and log in (`hf auth login`) before the first run.
"""
import argparse
import functools
import gc
import glob
import json
import math
import os
import random
import re
import sys
import time
import urllib.request

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_map
from PIL import Image, ImageDraw, ImageFont

MODEL_ID = "briaai/Fibo-2.0-edit-turbo"
#: the parts of the model repository this script reads
MODEL_FILES = ["model_index.json", "transformer/*", "text_encoder/*", "processor/*", "vae/*", "fonts/*"]
#: bria-vgl's "generate" instructions (the "thales" dialect): the system prompt, its field guide and the user turn
VGL_URL = "https://raw.githubusercontent.com/Bria-AI/vgl/dev/src/bria_vgl/dialects/thales"
VGL_FILES = {"system": "prompts/generate/system_prompt.txt", "fields": "input_schema/generate.txt",
             "user": "prompts/generate/final_prompt.txt"}
FONT_NAMES = ("Aileron-Regular.otf", "DejaVuSans.ttf")

# ── the pipeline's prompt forms (diffusers' BriaFibo2Pipeline) ─────────────────────────────────────────────────────────

#: where an image stands in a Qwen3-VL prompt, and the chat turn every prompt is wrapped in
VISION_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"
PROMPT_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
#: the Qwen3-VL hidden states behind the transformer's six text inputs: the Perceiver's, then one per injection block
TEXT_ENCODER_OUT_LAYERS = ((9, 20, 31), (5, 16, 27), (7, 18, 29), (9, 20, 31), (11, 22, 33), (13, 24, 35))
#: the null caption FIBO 2.0 was trained with: the negative prompt of classifier-free guidance (--guidance above 1)
NULL_CAPTION = ('{"short_description":"","objects":[],"background_setting":"","lighting":{},"aesthetics":{},'
                '"photographic_characteristics":{},"style_medium":"","context":"","artistic_style":""}')
#: a caption's text images join from this noise level down
TEXT_IMAGES_MAX_NOISE = 0.9
_SCORES = ("aesthetic_score", "preference_score")
#: the scheduler's exponential time shift, linear in the image's token count (scheduler/scheduler_config.json)
BASE_SEQ_LEN, MAX_SEQ_LEN, BASE_SHIFT, MAX_SHIFT = 256, 4096, 0.5, 1.15


def _empty(value):
    if isinstance(value, float):
        return math.isnan(value)
    return value is None or (isinstance(value, (str, dict, list)) and not value)


def _clean(value):
    if isinstance(value, dict):
        value = {k: _clean(v) for k, v in value.items()}
        return {k: v for k, v in value.items() if not _empty(v)}
    if isinstance(value, list):
        return [v for v in map(_clean, value) if not _empty(v)]
    return value


def parse_prompt(prompt):
    """(caption dict, or the prompt itself when it isn't a JSON object): empty values and scores dropped, the order kept."""
    try:
        caption = json.loads(prompt)
    except (TypeError, ValueError):
        return prompt
    if not isinstance(caption, dict):
        return prompt
    caption.pop("edit_instruction", None)
    if "structured_caption" in caption:
        caption = caption.pop("structured_caption")
        if not isinstance(caption, dict):
            return caption or None
    keys = set(caption)
    caption = {key: ({k: v for k, v in value.items() if k not in _SCORES} if isinstance(value, dict) else value)
               for key, value in _clean(caption).items() if key not in _SCORES}
    caption = {key: value for key, value in caption.items() if value != {}}
    if keys == {"short_description", "text_render"}:
        caption = {"short_description": caption.get("short_description") or "",
                   "text_render": caption.get("text_render") or []}
    return caption or None


def generation_prompt(prompt):
    """A caption cleaned and serialised compactly, as the pipeline encodes it; any other text as it is."""
    caption = parse_prompt(prompt)
    return prompt if isinstance(caption, str) else json.dumps(caption or {}, separators=(",", ":"), ensure_ascii=False)


def text_entries(caption):
    """The texts a caption asks to render (text_render entries with text), in order."""
    if not isinstance(caption, dict):
        return []
    return [e["text"] for e in (caption.get("text_render") or [])
            if isinstance(e, dict) and isinstance(e.get("text"), str) and e["text"]]


def mark_text_images(caption, marker):
    """Each text entry gets an "image_tokens" field: where its text image stands in the prompt."""
    for entry in caption.get("text_render") or []:
        if isinstance(entry, dict) and isinstance(entry.get("text"), str) and entry["text"]:
            entry["image_tokens"] = marker
    return caption


def get_edit_prompt(prompt: str, num_images: int, text_images: bool = False) -> str:
    """Format an edit prompt with up to 4 reference image markers following FIBO 2.0 specification."""
    instruction = None
    if isinstance(prompt, str):
        try:
            raw_obj = json.loads(prompt)
            if isinstance(raw_obj, dict):
                instruction = raw_obj.get("edit_instruction")
        except (json.JSONDecodeError, TypeError):
            pass
    elif isinstance(prompt, dict):
        instruction = prompt.get("edit_instruction")

    caption = parse_prompt(prompt)
    if instruction is None:
        instruction = prompt if isinstance(caption, str) else None

    if isinstance(instruction, str):
        # Sanitize FLUX.2 style referencing ("Image 1", "from Image 1") to FIBO 2.0 standards
        instruction_clean = re.sub(r"\bImage\s+(\d+)\b", r"<image_\1>", instruction, flags=re.IGNORECASE)
        if num_images <= 1:
            instruction_clean = re.sub(r"\s*(?:from|in|of)?\s*<image_1>\s*", " ", instruction_clean, flags=re.IGNORECASE).strip()
            instruction_clean = re.sub(r"\s+", " ", instruction_clean)
        instruction = instruction_clean

    if text_images and isinstance(caption, dict):
        mark_text_images(caption, VISION_MARKER)

    marker = VISION_MARKER
    if num_images <= 1:
        edit = {"image": marker}
    else:
        edit = {f"image_{k}": marker for k in range(1, num_images + 1)}

    if isinstance(caption, dict) and caption:
        edit["structured_caption"] = caption
    edit["edit_instruction"] = instruction or prompt
    return json.dumps(edit, separators=(",", ":"), ensure_ascii=False)


def grey_out(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Masked edit: replace the white region of the mask with flat mid-grey (128, 128, 128).

    White (>= 128) marks the region to regenerate; black (< 128) is kept untouched.
    Conforms to Bria FIBO 2.0 Edit Turbo specification.
    """
    img_rgb = image.convert("RGB")
    grey = Image.new("RGB", img_rgb.size, (128, 128, 128))
    binary_mask = (
        mask.resize(img_rgb.size, Image.Resampling.NEAREST)
        .convert("L")
        .point(lambda v: 255 if v >= 128 else 0)
    )
    return Image.composite(grey, img_rgb, binary_mask)


def calculate_mu(width, height):
    n = (width // 16) * (height // 16)
    m = (MAX_SHIFT - BASE_SHIFT) / (MAX_SEQ_LEN - BASE_SEQ_LEN)
    return n * m + (BASE_SHIFT - m * BASE_SEQ_LEN)


def sigmas_for(steps, width, height):
    """The pipeline's schedule: linspace(1, 1/N, N) under the exponential time shift, then 0."""
    e = math.exp(calculate_mu(width, height))
    return [e / (e + 1.0 / float(u) - 1.0) for u in np.linspace(1.0, 1.0 / steps, steps)] + [0.0]


# ── the images of a caption's text (diffusers' pipelines/bria_fibo2/text_images.py, Apache-2.0, Bria AI and The
#    HuggingFace Team): each text drawn in one line, black on white, as FIBO 2.0's training drew them ──────────────────

FONT_SIZE, MAX_LINE_CHARS, _PADDING, _LINE_SPACING = 20, 256, 2, 4


@functools.lru_cache(maxsize=None)
def load_font(path):
    return ImageFont.truetype(path, FONT_SIZE, layout_engine=ImageFont.Layout.BASIC)


@functools.lru_cache(maxsize=None)
def _default_font():
    font = ImageFont.load_default(FONT_SIZE)
    if not isinstance(font, ImageFont.FreeTypeFont):
        raise ImportError("Drawing a caption's text needs Pillow 10.1 or later, built with FreeType.")
    return font


@functools.lru_cache(maxsize=4096)
def _has_glyph(font, character):
    try:
        notdef, mask = font.getmask("￾"), font.getmask(character)
    except Exception:
        return True
    return not (mask.size == notdef.size and bytes(mask) == bytes(notdef))


def _font_runs(text, fonts):
    runs = []
    for character in text:
        font = fonts[0]
        if not (ord(character) < 128 or _has_glyph(fonts[0], character)):
            font = next((f for f in fonts[1:] if _has_glyph(f, character)), fonts[0])
        if runs and runs[-1][1] is font:
            runs[-1][0] += character
        else:
            runs.append([character, font])
    return [(run, font) for run, font in runs]


def _draw_one_font(lines, font):
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1), "white"))
    min_box = draw.textbbox((0, 0), "Ag", font=font)
    min_height = max(1, min_box[3] - min_box[1])
    sizes, width, height = [], 1, 0
    for line in lines:
        box = draw.textbbox((0, 0), line or " ", font=font)
        line_height = max(min_height, box[3] - box[1])
        sizes.append((box, line_height))
        width = max(width, box[2] - box[0])
        height += line_height
    height += _LINE_SPACING * (len(lines) - 1)
    image = Image.new("RGB", (width + _PADDING * 2, height + _PADDING * 2), "white")
    draw = ImageDraw.Draw(image)
    y = _PADDING
    for line, (box, line_height) in zip(lines, sizes):
        draw.text((_PADDING - box[0], y - box[1]), line, font=font, fill=(0, 0, 0))
        y += line_height + _LINE_SPACING
    return image


def _draw_mixed(lines, fonts):
    ascent, descent = fonts[0].getmetrics()
    line_height = max(1, ascent + descent)
    runs_per_line = [_font_runs(line or " ", fonts) for line in lines]
    width = max(max(1, round(sum(f.getlength(r) for r, f in runs))) for runs in runs_per_line)
    height = line_height * len(lines) + _LINE_SPACING * (len(lines) - 1)
    image = Image.new("RGB", (width + _PADDING * 2, height + _PADDING * 2), "white")
    draw = ImageDraw.Draw(image)
    y = _PADDING
    for runs in runs_per_line:
        x = float(_PADDING)
        for run, font in runs:
            draw.text((x, y + ascent), run, font=font, fill=(0, 0, 0), anchor="ls")
            x += font.getlength(run)
        y += line_height + _LINE_SPACING
    return image


def draw_text_image(text, fallback_fonts=()):
    font = _default_font()
    line = " ".join(text.split())
    lines = [line[i:i + MAX_LINE_CHARS] for i in range(0, len(line), MAX_LINE_CHARS)] or [""]
    if all(ord(c) < 128 or _has_glyph(font, c) for c in line):
        return _draw_one_font(lines, font)
    return _draw_mixed(lines, [font, *fallback_fonts])


# ── the transformer (diffusers' BriaFibo2Transformer2DModel, module for module: the checkpoint loads by its names) ────

ADALN_EMBED_DIM = 256
FP16_MAX = 65504.0


def compute_gist_count(text_lengths, min_gist, max_gist, gist_step, min_text_len, max_text_len):
    span = max_text_len - min_text_len
    out = []
    for n in text_lengths:
        ratio = min(1.0, max(0.0, (float(n) - min_text_len) / span))
        raw = min_gist + ratio * (max_gist - min_gist)
        out.append(max(min_gist, min(max_gist, int(math.ceil(raw / gist_step)) * gist_step)))
    return out


def grid_ids(size, start):
    axes = [np.arange(s0, s0 + n, dtype=np.int64) for s0, n in zip(start, size)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def rope_tables(axes_dims, axes_lens, theta):
    tables = []
    for d, n in zip(axes_dims, axes_lens):
        freqs = 1.0 / (theta ** (np.arange(0, d, 2, dtype=np.float64) / d))
        ang = np.outer(np.arange(n, dtype=np.float64), freqs).astype(np.float32)
        tables.append((np.cos(ang), np.sin(ang)))
    return tables


def apply_rope(x, cos, sin):
    """x (B, L, H, D), interleaved (real, imaginary) pairs, rotated in float32 and cast back."""
    B, L, H, D = x.shape
    xf = x.astype(mx.float32).reshape(B, L, H, D // 2, 2)
    xr, xi = xf[..., 0], xf[..., 1]
    out = mx.stack([xr * cos - xi * sin, xr * sin + xi * cos], axis=-1)
    return out.reshape(B, L, H, D).astype(x.dtype)


def to_stream(out, dtype):
    """A float16 module output back into the residual stream's dtype, clipped to float16's range."""
    return mx.clip(out, -FP16_MAX, FP16_MAX).astype(dtype)


def cast_fp16(module):
    module.update(tree_map(lambda v: v.astype(mx.float16) if isinstance(v, mx.array) and
                           mx.issubdtype(v.dtype, mx.floating) else v, module.parameters()))
    mx.eval(module.parameters())


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=256):
        super().__init__()
        mid_size = mid_size or out_size
        self.mlp = [nn.Linear(frequency_embedding_size, mid_size), nn.SiLU(), nn.Linear(mid_size, out_size)]
        self.frequency_embedding_size = frequency_embedding_size

    def __call__(self, t):
        half = self.frequency_embedding_size // 2
        freqs = mx.exp(-math.log(10000.0) * mx.arange(0, half, dtype=mx.float32) / half)
        args = t.astype(mx.float32)[:, None] * freqs[None]
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1).astype(self.mlp[0].weight.dtype)
        for layer in self.mlp:
            emb = layer(emb)
        return emb


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    def __init__(self, query_dim, heads, dim_head, kv_dim=None, eps=1e-5):
        super().__init__()
        kv_dim = kv_dim or query_dim
        self.heads, self.dim_head = heads, dim_head
        self.to_q = nn.Linear(query_dim, heads * dim_head, bias=False)
        self.to_k = nn.Linear(kv_dim, heads * dim_head, bias=False)
        self.to_v = nn.Linear(kv_dim, heads * dim_head, bias=False)
        self.to_out = [nn.Linear(heads * dim_head, query_dim, bias=False)]
        self.norm_q = nn.RMSNorm(dim_head, eps=eps)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps)

    def __call__(self, x, context=None, mask=None, rope=None):
        B, L, _ = x.shape
        ctx = x if context is None else context
        S = ctx.shape[1]
        q = self.norm_q(self.to_q(x).reshape(B, L, self.heads, self.dim_head))
        k = self.norm_k(self.to_k(ctx).reshape(B, S, self.heads, self.dim_head))
        v = self.to_v(ctx).reshape(B, S, self.heads, self.dim_head)
        if rope is not None:
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        o = mx.fast.scaled_dot_product_attention(q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3),
                                                 v.transpose(0, 2, 1, 3), scale=self.dim_head ** -0.5, mask=mask)
        return self.to_out[0](o.transpose(0, 2, 1, 3).reshape(B, L, -1))


class TransformerBlock(nn.Module):
    """Z-Image's block; with cross_attention_dim an injection block that also reads its Qwen3-VL bundle."""

    def __init__(self, dim, n_heads, norm_eps, cross_attention_dim=None):
        super().__init__()
        self.attention = Attention(dim, n_heads, dim // n_heads, eps=norm_eps)
        self.feed_forward = FeedForward(dim, (dim * 8 // 3) // 64 * 64)
        self.attention_norm1 = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = nn.RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = nn.RMSNorm(dim, eps=norm_eps)
        self.adaLN_modulation = [nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim)]
        self.has_cross = cross_attention_dim is not None
        if self.has_cross:
            self.cross_attn_norm1 = nn.RMSNorm(dim, eps=norm_eps)
            self.cross_attention = Attention(dim, n_heads, dim // n_heads, kv_dim=cross_attention_dim, eps=norm_eps)
            self.cross_attn_norm2 = nn.RMSNorm(dim, eps=norm_eps)
            self.cross_attn_gate = mx.zeros((1,))
        #: --precision float16: attention, cross-attention and the FFN compute in float16 (enable_fp16_compute)
        self.fp16 = False

    def _mod(self, emb):
        scale_msa, gate_msa, scale_mlp, gate_mlp = mx.split(self.adaLN_modulation[0](emb), 4, axis=-1)
        return 1.0 + scale_msa, mx.tanh(gate_msa), 1.0 + scale_mlp, mx.tanh(gate_mlp)

    def __call__(self, x, mask, rope, adaln=None, noise_mask=None, adaln_noisy=None, adaln_clean=None, text=None,
                 text_mask=None):
        if noise_mask is not None:
            sel = (noise_mask == 1)[:, :, None]
            noisy, clean = self._mod(adaln_noisy), self._mod(adaln_clean)
            scale_msa, gate_msa, scale_mlp, gate_mlp = (mx.where(sel, a[:, None, :], b[:, None, :])
                                                        for a, b in zip(noisy, clean))
        else:
            scale_msa, gate_msa, scale_mlp, gate_mlp = (m[:, None, :] for m in self._mod(adaln))
        fp16 = self.fp16
        y = self.attention_norm1(x) * scale_msa
        attn = self.attention(y.astype(mx.float16) if fp16 else y, mask=mask, rope=rope)
        x = x + gate_msa * self.attention_norm2(to_stream(attn, x.dtype) if fp16 else attn)
        if self.has_cross:
            y = self.cross_attn_norm1(x)
            cross = self.cross_attention(y.astype(mx.float16) if fp16 else y,
                                         context=text.astype(mx.float16) if fp16 else text, mask=text_mask)
            x = x + mx.tanh(self.cross_attn_gate) * self.cross_attn_norm2(to_stream(cross, x.dtype) if fp16 else cross)
        y = self.ffn_norm1(x) * scale_mlp
        f = self.feed_forward(y.astype(mx.float16) if fp16 else y)
        return x + gate_mlp * self.ffn_norm2(to_stream(f, x.dtype) if fp16 else f)


class PerceiverLayer(nn.Module):
    def __init__(self, dim, n_heads, cross_attention_dim, norm_eps):
        super().__init__()
        self.cross_attn_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.cross_attention = Attention(dim, n_heads, dim // n_heads, kv_dim=cross_attention_dim, eps=norm_eps)
        self.self_attn_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.self_attention = Attention(dim, n_heads, dim // n_heads, eps=norm_eps)
        self.ffn_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn = FeedForward(dim, (dim * 8 // 3) // 64 * 64)

    def __call__(self, latents, text, text_mask=None, mask=None):
        latents = latents + self.cross_attention(self.cross_attn_norm(latents), context=text, mask=text_mask)
        latents = latents + self.self_attention(self.self_attn_norm(latents), mask=mask)
        return latents + self.ffn(self.ffn_norm(latents))


class PerceiverResampler(nn.Module):
    def __init__(self, dim, n_heads, cross_attention_dim, num_queries, num_layers, norm_eps):
        super().__init__()
        self.latent_queries = mx.zeros((num_queries, dim))
        self.layers = [PerceiverLayer(dim, n_heads, cross_attention_dim, norm_eps) for _ in range(num_layers)]
        self.norm_out = nn.RMSNorm(dim, eps=norm_eps)

    def __call__(self, text, num_gist_tokens, text_mask=None, mask=None):
        B = text.shape[0]
        latents = mx.broadcast_to(self.latent_queries[:num_gist_tokens][None],
                                  (B, num_gist_tokens, self.latent_queries.shape[-1]))
        for layer in self.layers:
            latents = layer(latents, text, text_mask, mask)
        return self.norm_out(latents)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6, affine=False)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.adaLN_modulation = [nn.SiLU(), nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size)]

    def __call__(self, x, c):
        scale = 1.0 + self.adaLN_modulation[1](nn.silu(c))
        return self.linear(self.norm_final(x) * scale[:, None, :])


class Fibo2Transformer(nn.Module):
    """__call__(latents (B, C, H, W), timestep (B,) in [0, 1], text bundles (B, 6, S, 7680), context latents or None)
    -> the velocity (B, C, H, W) in float32."""

    def __init__(self, c):
        super().__init__()
        dim, p, cin = int(c["dim"]), int(c["patch_size"]), int(c["in_channels"])
        n_heads, eps = int(c["n_heads"]), float(c["norm_eps"])
        self.patch_size, self.in_channels = p, cin
        self.t_scale = float(c.get("t_scale", 1000.0))
        self.max_gist = int(c["max_num_gist_tokens"])
        self.gist_args = (int(c["min_num_gist_tokens"]), int(c["max_num_gist_tokens"]), int(c["gist_step"]),
                          int(c["gist_min_text_len"]), int(c["gist_max_text_len"]))
        self.injection = [int(i) for i in c["injection_layer_ids"]]
        self.x_embedder = nn.Linear(p * p * cin, dim)
        self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
        self.noise_refiner = [TransformerBlock(dim, n_heads, eps) for _ in range(int(c["n_refiner_layers"]))]
        self.perceiver = PerceiverResampler(dim, n_heads, int(c["cap_feat_dim"]), self.max_gist,
                                            int(c["perceiver_num_layers"]), eps)
        self.layers = [TransformerBlock(dim, n_heads, eps, int(c["cap_feat_dim"]) if i in self.injection else None)
                       for i in range(int(c["n_layers"]))]
        self.final_layer = FinalLayer(dim, p * p * cin)
        self._rope = rope_tables([int(v) for v in c["axes_dims"]], [int(v) for v in c["axes_lens"]],
                                 float(c["rope_theta"]))

    def _rope_for(self, ids):
        cos = np.concatenate([tab[0][ids[:, i]] for i, tab in enumerate(self._rope)], axis=-1)
        sin = np.concatenate([tab[1][ids[:, i]] for i, tab in enumerate(self._rope)], axis=-1)
        return mx.array(cos)[None, :, None, :], mx.array(sin)[None, :, None, :]

    def _patchify(self, lat):
        B, C, H, W = lat.shape
        p = self.patch_size
        x = lat.reshape(B, C, H // p, p, W // p, p).transpose(0, 2, 4, 1, 3, 5)
        return x.reshape(B, (H // p) * (W // p), C * p * p)

    def __call__(self, latents, timestep, text, context=None):
        B, C, H, W = latents.shape
        p = self.patch_size
        Ht, Wt = H // p, W // p
        n_img = Ht * Wt
        dtype = self.x_embedder.weight.dtype
        adaln_noisy = self.t_embedder(timestep * self.t_scale)
        adaln_clean = self.t_embedder(mx.zeros_like(timestep))
        # the image on RoPE plane max_gist + 1, context image k on the k-th plane after it
        plane = self.max_gist + 1
        tokens, ids = [self._patchify(latents.astype(dtype))], [grid_ids((1, Ht, Wt), (plane, 0, 0))]
        for k, lat in enumerate(context or []):
            tokens.append(self._patchify(lat.astype(dtype)))
            ids.append(grid_ids((1, lat.shape[2] // p, lat.shape[3] // p), (plane + k + 1, 0, 0)))
        h = self.x_embedder(mx.concatenate(tokens, axis=1))
        img_cos, img_sin = self._rope_for(np.concatenate(ids, axis=0))
        n_all = h.shape[1]
        refiner_mask = None
        if context:
            refiner_mask = mx.concatenate([mx.ones((B, n_img), dtype=mx.int32),
                                           mx.zeros((B, n_all - n_img), dtype=mx.int32)], axis=1)
        for blk in self.noise_refiner:
            h = blk(h, None, (img_cos, img_sin), adaln=adaln_noisy, noise_mask=refiner_mask, adaln_noisy=adaln_noisy,
                    adaln_clean=adaln_clean)
        # the gist tokens: their count from the text length, the Perceiver over the first bundle
        n_gist = compute_gist_count([text.shape[2]] * B, *self.gist_args)[0]
        gist = self.perceiver(text[:, 0], n_gist)
        g_cos, g_sin = self._rope_for(grid_ids((n_gist, 1, 1), (1, 0, 0)))
        h = mx.concatenate([gist, h], axis=1)
        rope = (mx.concatenate([g_cos, img_cos], axis=1), mx.concatenate([g_sin, img_sin], axis=1))
        noise_mask = np.zeros((B, h.shape[1]), dtype=np.int32)
        noise_mask[:, n_gist:n_gist + n_img] = 1
        noise_mask = mx.array(noise_mask)
        bundle = {layer_id: k + 1 for k, layer_id in enumerate(self.injection)}
        for i, blk in enumerate(self.layers):
            h = blk(h, None, rope, noise_mask=noise_mask, adaln_noisy=adaln_noisy, adaln_clean=adaln_clean,
                    text=text[:, bundle[i]] if i in bundle else None)
        out = self.final_layer(h[:, n_gist:n_gist + n_img], adaln_noisy)
        out = out.reshape(B, Ht, Wt, C, p, p).transpose(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
        return out.astype(mx.float32)


def enable_fp16_compute(model):
    """--precision float16: attention, cross-attention and the FFN of every block compute in float16; the norms, the
    modulation, the residual stream, the Perceiver and the embedders keep the weights' dtype."""
    for blk in list(model.noise_refiner) + list(model.layers):
        cast_fp16(blk.attention)
        cast_fp16(blk.feed_forward)
        if blk.has_cross:
            cast_fp16(blk.cross_attention)
        blk.fp16 = True


def load_transformer(folder, dtype):
    with open(os.path.join(folder, "config.json")) as f:
        model = Fibo2Transformer(json.load(f))
    weights = {}
    for path in sorted(glob.glob(os.path.join(folder, "*.safetensors"))):
        weights.update(mx.load(path))
    want = dict(tree_flatten(model.parameters()))
    missing = sorted(set(want) - set(weights))
    bad = sorted(k for k in set(want) & set(weights) if tuple(want[k].shape) != tuple(weights[k].shape))
    if missing or bad:
        raise ValueError(f"{folder}: not FIBO 2.0's transformer (missing {missing[:3]}, shapes {bad[:3]})")
    model.load_weights([(k, v.astype(dtype)) for k, v in weights.items() if k in want], strict=True)
    mx.eval(model.parameters())
    return model


# ── the text encoder: Qwen3-VL-4B-Instruct through mlx-vlm ──────────────────────────────────────────────────────────────

class TextEncoder:
    """The model's text_encoder/ in mlx-vlm's Qwen3-VL, the tokenizer from processor/. encode(): the transformer's six
    bundles (1, 6, L, 7680), every layer scaled to unit RMS per token in float32; write(): a chat reply."""

    def __init__(self, folder, processor_dir, dtype):
        from mlx_vlm.models.qwen3_vl.config import ModelConfig, TextConfig, VisionConfig
        from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLImageProcessor
        from mlx_vlm.models.qwen3_vl.qwen3_vl import Model
        from transformers import AutoTokenizer
        with open(os.path.join(folder, "config.json")) as f:
            cfg = json.load(f)
        tcfg = dict(cfg["text_config"])
        rp = tcfg.get("rope_parameters")
        if rp and "rope_theta" not in tcfg:          # transformers 5 writes one rope block; mlx-vlm reads flat fields
            tcfg["rope_theta"] = rp.get("rope_theta", 5000000)
            tcfg["rope_scaling"] = {"mrope_section": rp.get("mrope_section", [24, 20, 20]),
                                    "rope_type": rp.get("rope_type", "default"),
                                    "mrope_interleaved": rp.get("mrope_interleaved", False)}
        vcfg = dict(cfg["vision_config"])
        if vcfg.get("model_type") == "qwen3_vl_vision":
            vcfg["model_type"] = "qwen3_vl"
        tc, vc = TextConfig.from_dict(tcfg), VisionConfig.from_dict(vcfg)
        config = ModelConfig(text_config=tc, vision_config=vc, model_type=cfg.get("model_type", "qwen3_vl"),
                             image_token_id=cfg.get("image_token_id", 151655),
                             video_token_id=cfg.get("video_token_id", 151656),
                             vision_start_token_id=cfg.get("vision_start_token_id", 151652),
                             vision_end_token_id=cfg.get("vision_end_token_id", 151653),
                             vocab_size=int(tc.vocab_size))
        self.model = Model(config)
        raw = {}
        for path in sorted(glob.glob(os.path.join(folder, "*.safetensors"))):
            raw.update(mx.load(path))
        weights = self.model.vision_tower.sanitize(self.model.sanitize(raw))
        want = {k for k, _ in tree_flatten(self.model.parameters())}
        missing = sorted(want - set(weights))
        if missing:
            raise ValueError(f"{folder}: {len(missing)} text-encoder parameters missing ({missing[:3]})")
        self.model.load_weights([(k, v.astype(dtype)) for k, v in weights.items() if k in want], strict=True)
        mx.eval(self.model.parameters())
        del raw, weights
        self.tokenizer = AutoTokenizer.from_pretrained(processor_dir)
        with open(os.path.join(processor_dir, "processor_config.json")) as f:
            ip = json.load(f).get("image_processor", {})
        size = ip.get("size") or {}
        self.merge = int(cfg["vision_config"].get("spatial_merge_size", 2))
        self.images = Qwen3VLImageProcessor(
            patch_size=int(ip.get("patch_size", 16)), temporal_patch_size=int(ip.get("temporal_patch_size", 2)),
            merge_size=self.merge, min_pixels=int(size.get("shortest_edge", 3136)),
            max_pixels=int(size.get("longest_edge", 1605632)), image_mean=ip.get("image_mean", [0.5, 0.5, 0.5]),
            image_std=ip.get("image_std", [0.5, 0.5, 0.5]))
        self.image_token_id = int(config.image_token_index)
        vocab = self.tokenizer.get_vocab()
        self.stop_ids = {vocab[t] for t in ("<|im_end|>", "<|endoftext|>") if t in vocab}
        self.dtype = dtype

    def _inputs(self, text, images=None):
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if not images:
            return mx.array(np.array(ids, dtype=np.int32))[None], None, None
        out = self.images.preprocess(images=[im.convert("RGB") for im in images])
        thw = np.array(out["image_grid_thw"])
        counts = [int(np.prod(row)) // (self.merge * self.merge) for row in thw]
        expanded, k = [], 0
        for t in ids:                          # each image's one <|image_pad|> -> its merged patch count
            if t == self.image_token_id:
                expanded.extend([self.image_token_id] * counts[k])
                k += 1
            else:
                expanded.append(t)
        return (mx.array(np.array(expanded, dtype=np.int32))[None], mx.array(np.array(out["pixel_values"])),
                mx.array(thw))

    def encode(self, prompt, images=None, text_images=None):
        """The pipeline's prompt embeddings: (1, 6, L, 7680).
        images: list of up to 4 reference PIL Images for multimodal conditioning.
        text_images: list of rendered text images.
        """
        from mlx_vlm.models.base import create_attention_mask
        all_images = []
        if images:
            ref_list = images if isinstance(images, list) else [images]
            all_images.extend([im.convert("RGB") for im in ref_list[:4]])
            prompt = get_edit_prompt(prompt, len(all_images), text_images=bool(text_images))
            if text_images:
                all_images.extend(text_images)
        elif text_images:
            try:
                caption_obj = json.loads(prompt)
                prompt = json.dumps(mark_text_images(caption_obj, VISION_MARKER), separators=(",", ":"),
                                    ensure_ascii=False)
            except (json.JSONDecodeError, TypeError):
                pass
            all_images.extend(text_images)
        else:
            prompt = generation_prompt(prompt)

        ids, pv, thw = self._inputs(PROMPT_TEMPLATE.format(prompt), all_images or None)
        feats = self.model.get_input_embeddings(ids, pixel_values=pv, image_grid_thw=thw)
        lm = self.model.language_model.model
        pos, h = feats.position_ids, feats.inputs_embeds
        mask = create_attention_mask(h, None)
        rope = None
        if pos is not None and not lm.layers[0].self_attn.rotary_emb.fused_apply:
            rope = lm.layers[0].self_attn.rotary_emb(h, pos)
        deep = getattr(feats, "deepstack_visual_embeds", None)
        wanted = sorted({i for b in TEXT_ENCODER_OUT_LAYERS for i in b})
        states = {}
        for i, layer in enumerate(lm.layers[:max(wanted)]):
            h = layer(h, mask, None, pos, rope)
            if deep is not None and i < len(deep):
                h = lm._deepstack_process(h, feats.visual_pos_masks, deep[i])
            if i + 1 in wanted:                    # hidden_states[i + 1]: after layer i
                s = h.astype(mx.float32)
                states[i + 1] = (s * mx.rsqrt(mx.mean(s * s, axis=-1, keepdims=True) + 1e-6)).astype(self.dtype)
        out = mx.stack([mx.concatenate([states[i] for i in b], axis=-1) for b in TEXT_ENCODER_OUT_LAYERS], axis=1)
        mx.eval(out)
        return out

    def write(self, system, user, sample_seed=None, max_tokens=2048):
        """The model's reply to a system + user chat: greedy, or sampled (0.8 / top-p 0.9) from sample_seed."""
        from mlx_vlm.generate import generate_step
        chat = (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
                f"<|im_start|>assistant\n")
        ids, _, _ = self._inputs(chat)
        if sample_seed is not None:
            mx.random.seed(int(sample_seed))
        out = []
        for tok, _ in generate_step(ids, self.model, None, None, max_tokens=max_tokens,
                                    temperature=0.8 if sample_seed is not None else 0.0,
                                    top_p=0.9 if sample_seed is not None else 1.0):
            t = int(tok.item() if hasattr(tok, "item") else tok)
            if t in self.stop_ids:
                break
            out.append(t)
        return self.tokenizer.decode(out, skip_special_tokens=True)


# ── the FLUX.2 VAE's decoder: its layers in MLX, its weights from the model's vae/ ─────────────────────────────────────

def _group_norm(ch):
    return nn.GroupNorm(32, ch, eps=1e-6, pytorch_compatible=True)


class ResnetBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.norm1, self.conv1 = _group_norm(cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.norm2, self.conv2 = _group_norm(cout), nn.Conv2d(cout, cout, 3, padding=1)
        if cin != cout:
            self.conv_shortcut = nn.Conv2d(cin, cout, 1)

    def __call__(self, x):
        h = self.conv2(nn.silu(self.norm2(self.conv1(nn.silu(self.norm1(x))))))
        return (self.conv_shortcut(x) if "conv_shortcut" in self else x) + h


class MidAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.group_norm = _group_norm(ch)
        self.to_q, self.to_k, self.to_v = nn.Linear(ch, ch), nn.Linear(ch, ch), nn.Linear(ch, ch)
        self.to_out = [nn.Linear(ch, ch)]

    def __call__(self, x):
        B, H, W, C = x.shape
        h = self.group_norm(x).reshape(B, H * W, C)
        q, k, v = (f(h)[:, None] for f in (self.to_q, self.to_k, self.to_v))
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=C ** -0.5)[:, 0]
        return x + self.to_out[0](o).reshape(B, H, W, C)


class MidBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.resnets = [ResnetBlock(ch, ch), ResnetBlock(ch, ch)]
        self.attentions = [MidAttention(ch)]

    def __call__(self, x):
        return self.resnets[1](self.attentions[0](self.resnets[0](x)))


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def __call__(self, x):
        return self.conv(mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2))


class UpBlock(nn.Module):
    def __init__(self, cin, cout, n, upsample):
        super().__init__()
        self.resnets = [ResnetBlock(cin if i == 0 else cout, cout) for i in range(n)]
        if upsample:
            self.upsamplers = [Upsample(cout)]

    def __call__(self, x):
        for r in self.resnets:
            x = r(x)
        return self.upsamplers[0](x) if "upsamplers" in self else x


class Decoder(nn.Module):
    """AutoencoderKLFlux2's decoder (diffusers' Decoder), channels last: (B, h, w, 32) -> (B, 8h, 8w, 3) in [-1, 1]."""

    def __init__(self, latent_channels=32, block_out_channels=(128, 256, 512, 512), layers_per_block=2):
        super().__init__()
        rev = list(reversed(block_out_channels))
        self.conv_in = nn.Conv2d(latent_channels, rev[0], 3, padding=1)
        self.mid_block = MidBlock(rev[0])
        self.up_blocks, cin = [], rev[0]
        for i, cout in enumerate(rev):
            self.up_blocks.append(UpBlock(cin, cout, layers_per_block + 1, upsample=i < len(rev) - 1))
            cin = cout
        self.conv_norm_out = _group_norm(block_out_channels[0])
        self.conv_out = nn.Conv2d(block_out_channels[0], 3, 3, padding=1)

    def __call__(self, z):
        x = self.mid_block(self.conv_in(z))
        for blk in self.up_blocks:
            x = blk(x)
        return self.conv_out(nn.silu(self.conv_norm_out(x)))


class VaeDecoder:
    """The model's vae/: decode(latent) -> PIL. The latent the transformer works on is normalised with the VAE's batch-norm
    statistics on its 2x2-packed grid: undone first, then the post-quant conv and the decoder."""

    def __init__(self, folder, dtype):
        with open(os.path.join(folder, "config.json")) as f:
            cfg = json.load(f)
        widths = cfg.get("decoder_block_out_channels") or cfg["block_out_channels"]
        self.decoder = Decoder(int(cfg["latent_channels"]), tuple(widths), int(cfg["layers_per_block"]))
        self.post_quant_conv = nn.Conv2d(int(cfg["latent_channels"]), int(cfg["latent_channels"]), 1)
        raw = {}
        for path in sorted(glob.glob(os.path.join(folder, "*.safetensors"))):
            raw.update(mx.load(path))
        conv = lambda v: v.transpose(0, 2, 3, 1) if v.ndim == 4 else v   # (O, I, kh, kw) -> MLX's (O, kh, kw, I)
        dec = [(k[len("decoder."):], conv(v).astype(dtype)) for k, v in raw.items() if k.startswith("decoder.")]
        want = {k for k, _ in tree_flatten(self.decoder.parameters())}
        missing = sorted(want - {k for k, _ in dec})
        if missing:
            raise ValueError(f"{folder}: not a FLUX.2 VAE (decoder parameters missing: {missing[:3]})")
        self.decoder.load_weights(dec, strict=True)
        self.post_quant_conv.load_weights([(k[len("post_quant_conv."):], conv(v).astype(dtype))
                                           for k, v in raw.items() if k.startswith("post_quant_conv.")], strict=True)
        eps = float(cfg.get("batch_norm_eps", 1e-4))
        self.bn_mean = raw["bn.running_mean"].astype(mx.float32).reshape(1, -1, 1, 1)
        self.bn_std = mx.sqrt(raw["bn.running_var"].astype(mx.float32).reshape(1, -1, 1, 1) + eps)
        self.dtype = dtype
        mx.eval(self.decoder.parameters(), self.post_quant_conv.parameters(), self.bn_mean, self.bn_std)

    def decode(self, latents):
        B, C, H, W = latents.shape
        grid = latents.astype(mx.float32).reshape(B, C, H // 2, 2, W // 2, 2).transpose(0, 1, 3, 5, 2, 4)
        grid = grid.reshape(B, C * 4, H // 2, W // 2) * self.bn_std + self.bn_mean
        z = grid.reshape(B, C, 2, 2, H // 2, W // 2).transpose(0, 1, 4, 2, 5, 3).reshape(B, C, H, W)
        z = self.post_quant_conv(z.transpose(0, 2, 3, 1).astype(self.dtype))
        x = self.decoder(z)
        rgb = mx.clip(x[0].astype(mx.float32) / 2 + 0.5, 0, 1) * 255
        mx.eval(rgb)
        return Image.fromarray(np.array(rgb).round().astype(np.uint8))


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=0)

    def __call__(self, x):
        x = mx.pad(x, ((0, 0), (0, 1), (0, 1), (0, 0)))
        return self.conv(x)


class DownBlock(nn.Module):
    def __init__(self, cin, cout, n, downsample):
        super().__init__()
        self.resnets = [ResnetBlock(cin if i == 0 else cout, cout) for i in range(n)]
        if downsample:
            self.downsamplers = [Downsample(cout)]

    def __call__(self, x):
        for r in self.resnets:
            x = r(x)
        return self.downsamplers[0](x) if "downsamplers" in self else x


class Encoder(nn.Module):
    """AutoencoderKLFlux2's encoder, channels last: (B, H, W, 3) in [-1, 1] -> (B, H//8, W//8, 64)."""

    def __init__(self, latent_channels=32, block_out_channels=(128, 256, 512, 512), layers_per_block=2):
        super().__init__()
        self.conv_in = nn.Conv2d(3, block_out_channels[0], 3, padding=1)
        self.down_blocks = []
        cin = block_out_channels[0]
        for i, cout in enumerate(block_out_channels):
            self.down_blocks.append(DownBlock(cin, cout, layers_per_block, downsample=i < len(block_out_channels) - 1))
            cin = cout
        self.mid_block = MidBlock(block_out_channels[-1])
        self.conv_norm_out = _group_norm(block_out_channels[-1])
        self.conv_out = nn.Conv2d(block_out_channels[-1], 2 * latent_channels, 3, padding=1)

    def __call__(self, x):
        x = self.conv_in(x)
        for blk in self.down_blocks:
            x = blk(x)
        x = self.mid_block(x)
        return self.conv_out(nn.silu(self.conv_norm_out(x)))


class VaeEncoder:
    """The model's vae/: encode(PIL) -> normalized latents (1, 32, H//8, W//8)."""

    def __init__(self, folder, dtype):
        with open(os.path.join(folder, "config.json")) as f:
            cfg = json.load(f)
        widths = cfg.get("block_out_channels", [128, 256, 512, 512])
        latent_channels = int(cfg.get("latent_channels", 32))
        self.encoder = Encoder(latent_channels, tuple(widths), int(cfg.get("layers_per_block", 2)))
        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)

        raw = {}
        for path in sorted(glob.glob(os.path.join(folder, "*.safetensors"))):
            raw.update(mx.load(path))

        conv = lambda v: v.transpose(0, 2, 3, 1) if v.ndim == 4 else v   # (O, I, kh, kw) -> MLX's (O, kh, kw, I)
        enc = [(k[len("encoder."):], conv(v).astype(dtype)) for k, v in raw.items() if k.startswith("encoder.")]
        want = {k for k, _ in tree_flatten(self.encoder.parameters())}
        missing = sorted(want - {k for k, _ in enc})
        if missing:
            raise ValueError(f"{folder}: VAE encoder parameters missing: {missing[:3]}")
        self.encoder.load_weights(enc, strict=True)
        self.quant_conv.load_weights([(k[len("quant_conv."):], conv(v).astype(dtype))
                                      for k, v in raw.items() if k.startswith("quant_conv.")], strict=True)

        eps = float(cfg.get("batch_norm_eps", 1e-4))
        self.bn_mean = raw["bn.running_mean"].astype(mx.float32).reshape(1, -1, 1, 1)
        self.bn_std = mx.sqrt(raw["bn.running_var"].astype(mx.float32).reshape(1, -1, 1, 1) + eps)
        self.dtype = dtype
        mx.eval(self.encoder.parameters(), self.quant_conv.parameters(), self.bn_mean, self.bn_std)

    def encode(self, image: Image.Image, target_width: int, target_height: int) -> mx.array:
        """Preprocesses image, encodes via VAE, and normalizes into the DiT packed-latent space."""
        img = image.convert("RGB").resize((target_width, target_height), Image.Resampling.LANCZOS)
        arr = (np.array(img, dtype=np.float32) / 127.5) - 1.0  # (H, W, 3) in [-1, 1]
        x = mx.array(arr[None], dtype=self.dtype)              # (1, H, W, 3)

        # Forward pass through encoder + quant_conv
        h = self.encoder(x)
        h = self.quant_conv(h)                                 # (1, H//8, W//8, 64)

        # Extract mode/mean (first 32 channels)
        mean, _ = mx.split(h, 2, axis=-1)                      # (1, H//8, W//8, 32)
        z = mean.transpose(0, 3, 1, 2)                         # (1, 32, H//8, W//8)

        # Normalize with VAE BatchNorm statistics in 2x2 packed grid
        B, C, H, W = z.shape
        grid = z.reshape(B, C, H // 2, 2, W // 2, 2).transpose(0, 1, 3, 5, 2, 4)
        grid = grid.reshape(B, C * 4, H // 2, W // 2)
        grid_norm = (grid.astype(mx.float32) - self.bn_mean) / self.bn_std
        latents = grid_norm.reshape(B, C, 2, 2, H // 2, W // 2).transpose(0, 1, 4, 2, 5, 3).reshape(B, C, H, W)
        latents = latents.astype(self.dtype)
        mx.eval(latents)
        return latents


def load_encoder(folder, dtype):
    """The VAE encoder for reference image latent conditioning."""
    return VaeEncoder(os.path.join(folder, "vae"), dtype)


def load_decoder(folder, dtype):
    """The VAE decoder; fibo2_edit_turbo_t2i_mflux.py replaces it with mflux's."""
    return VaeDecoder(os.path.join(folder, "vae"), dtype)


# ── the run ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

def seed_noise(shape, seed):
    """The starting noise for `seed`: torch's CPU generator when torch is installed (the diffusers script's noise, so a
    seed gives the same image there), MLX's own generator otherwise."""
    try:
        import torch
    except ImportError:
        return mx.random.normal(shape, key=mx.random.key(int(seed)))
    gen = torch.Generator("cpu").manual_seed(int(seed))
    return mx.array(torch.randn(tuple(shape), generator=gen, dtype=torch.float32).numpy())


def vgl_prompts(cache_dir):
    """(system prompt, user template): bria-vgl's generate prompts, downloaded once into cache_dir."""
    os.makedirs(cache_dir, exist_ok=True)
    text = {}
    for key, rel in VGL_FILES.items():
        path = os.path.join(cache_dir, rel.replace("/", "__"))
        if not os.path.isfile(path):
            with urllib.request.urlopen(f"{VGL_URL}/{rel}", timeout=60) as r:
                data = r.read().decode("utf-8")
            with open(path, "w", encoding="utf-8") as f:
                f.write(data)
        with open(path, encoding="utf-8") as f:
            text[key] = f.read()
    system = text["system"].replace("{style_preset_block}", "").replace("{llm_input_schema}", text["fields"])
    return system, text["user"]


def extract_json(reply):
    """The first JSON object in a model reply (code fences and <think> blocks removed), or None."""
    t = re.sub(r"<think>.*?</think>", "", reply or "", flags=re.S).strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, flags=re.S)
    if m:
        t = m.group(1).strip()
    start = t.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(t[start:i + 1])
                except ValueError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def read_batch(path):
    """[(name, prompt)] from a JSON list (strings, or objects with 'prompt' and an optional 'slug' or 'name') or a text
    file with one prompt per line (blank lines and lines starting with # skipped)."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    try:
        items = json.loads(text)
    except ValueError:
        items = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    jobs = []
    for i, item in enumerate(items, 1):
        if isinstance(item, dict):
            jobs.append((str(item.get("slug") or item.get("name") or f"{i:02d}"), item["prompt"]))
        else:
            jobs.append((f"{i:02d}", str(item)))
    return jobs


def model_folder(model):
    """A local copy of the model repository (downloaded once into the Hugging Face cache), or the folder given."""
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download
    return snapshot_download(model, allow_patterns=MODEL_FILES)


def main(load_decoder=load_decoder, description="Text-to-image with FIBO 2.0 Edit Turbo on Apple silicon (MLX)"):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("prompt", nargs="?", help="what to generate (a plain description)")
    ap.add_argument("--json", help="a finished VGL caption: a JSON file holding the caption object, or an object with "
                                   "a 'caption' key (like assets/lake-pier.json)")
    ap.add_argument("--prompts", help="a batch: a JSON list of prompts (strings, or objects with 'prompt' and an "
                                      "optional 'slug' used as the file name) or a text file with one prompt per line")
    ap.add_argument("--out-dir", default=".", help="where a batch's images and captions are saved (default: here)")
    ap.add_argument("--plain", action="store_true", help="send the prompt as typed instead of writing a VGL caption")
    ap.add_argument("--sample", nargs="?", type=int, const=-1, default=None, metavar="CAPTION_SEED",
                    help="write the caption by sampling (temperature 0.8, top-p 0.9) instead of greedily: another take "
                         "on the same prompt, from CAPTION_SEED (default: the image's seed); saved in the JSON")
    ap.add_argument("--model", default=MODEL_ID, help=f"Hub id or local folder (default {MODEL_ID})")
    ap.add_argument("--width", type=int, default=1024, help="multiple of 16, about 1 megapixel in total")
    ap.add_argument("--height", type=int, default=1024, help="multiple of 16, about 1 megapixel in total")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=1.0, help="1.0 = no classifier-free guidance (Turbo's setting)")
    ap.add_argument("--seed", type=int, default=None, help="random for each image when omitted")
    ap.add_argument("--out", default=None, help="output PNG of a single run (default fibo2_edit_turbo_t2i_<seed>.png)")
    ap.add_argument("--precision", choices=["float16", "bfloat16", "float32"], default="float16",
                    help="float16 (default): attention and the feed-forward layers compute in float16, the rest in "
                         "bfloat16, about a quarter faster; bfloat16: the model's own precision; float32: twice the "
                         "memory")
    ap.add_argument("--cache", default=os.path.join(os.path.expanduser("~"), ".cache", "fibo2-vgl"),
                    help="where bria-vgl's prompt files are cached")
    a = ap.parse_args()
    if not (a.prompt or a.json or a.prompts):
        ap.error("give a prompt, --json with a VGL caption, or --prompts with a batch")
    width, height = a.width // 16 * 16, a.height // 16 * 16
    if (width, height) != (a.width, a.height):
        print(f"size rounded down to multiples of 16: {width}x{height}")
    dtype = mx.float32 if a.precision == "float32" else mx.bfloat16

    # the jobs: (output PNG, prompt, caption or None)
    if a.prompts:
        os.makedirs(a.out_dir, exist_ok=True)
        jobs = [(os.path.join(a.out_dir, name + ".png"), prompt, None) for name, prompt in read_batch(a.prompts)]
    elif a.json:
        with open(a.json, encoding="utf-8") as f:
            data = json.load(f)
        caption = data["caption"] if isinstance(data.get("caption"), dict) else data
        jobs = [(a.out, data.get("prompt") if isinstance(data.get("caption"), dict) else None, caption)]
    else:
        jobs = [(a.out, a.prompt, None)]
    todo = [list(j) for j in jobs if not (j[0] and os.path.isfile(j[0]))]
    if len(todo) < len(jobs):
        print(f"{len(jobs) - len(todo)} image(s) already there, skipped")
    if not todo:
        return 0
    folder = model_folder(a.model)
    fonts = [load_font(p) for p in (os.path.join(folder, "fonts", n) for n in FONT_NAMES) if os.path.isfile(p)]

    # one stage at a time, each alone in memory: 1. the text encoder (writes and embeds every prompt), 2. the transformer
    # (denoises every image), 3. the VAE (decodes them). At 1024x1024 the largest is the transformer's, about 17 GB
    # 1. the text encoder writes the captions and embeds them, then leaves before the transformer loads
    t0 = time.perf_counter()
    te = TextEncoder(os.path.join(folder, "text_encoder"), os.path.join(folder, "processor"), dtype)
    print(f"loaded the text encoder from {a.model} in {time.perf_counter() - t0:.0f} s")
    system = user_template = None
    negative = te.encode(NULL_CAPTION) if a.guidance > 1.0 else None
    runs = []
    for k, (out, prompt, caption) in enumerate(todo, 1):
        tag = f"[{k}/{len(todo)}] " if len(todo) > 1 else ""
        seed = a.seed if a.seed is not None else random.randrange(2 ** 31)
        caption_seed = None
        if caption is None and not a.plain:
            if system is None:
                system, user_template = vgl_prompts(a.cache)
            if a.sample is not None:
                caption_seed = seed if a.sample < 0 else a.sample
            t0 = time.perf_counter()
            user = user_template.replace("{user_prompt}", " ".join(prompt.split()))
            caption = extract_json(te.write(system, user, sample_seed=caption_seed))
            if caption is None:
                print(f"{tag}no usable caption was written: the prompt is used as typed")
            else:
                print(f"{tag}caption written in {time.perf_counter() - t0:.0f} s: "
                      f"{caption.get('short_description', '')[:120]}")
        text = generation_prompt(json.dumps(caption, ensure_ascii=False) if caption else prompt)
        parsed = parse_prompt(text)
        drawn = [draw_text_image(t, fonts) for t in text_entries(parsed)]
        cond = te.encode(text, drawn or None)
        # a caption with text images shows them from noise 0.9 down: before that, its entries' markers left empty
        base = te.encode(json.dumps(mark_text_images(json.loads(text), ""), separators=(",", ":"),
                                    ensure_ascii=False)) if drawn else None
        runs.append((out, prompt, caption, caption_seed, seed, cond, base))
    del te
    gc.collect()
    mx.clear_cache()

    # 2. the transformer denoises every image and leaves; only the final latents (1 MB each) are kept
    t0 = time.perf_counter()
    tr = load_transformer(os.path.join(folder, "transformer"), dtype)
    if a.precision == "float16":
        enable_fp16_compute(tr)
    print(f"loaded the transformer in {time.perf_counter() - t0:.0f} s ({a.precision})")
    sigmas = sigmas_for(a.steps, width, height)
    finals = []
    for k, (out, prompt, caption, caption_seed, seed, cond, base) in enumerate(runs, 1):
        tag = f"[{k}/{len(runs)}] " if len(runs) > 1 else ""
        t0 = time.perf_counter()
        lat = seed_noise((1, tr.in_channels, height // 8, width // 8), seed).astype(mx.float32)
        for s, s_next in zip(sigmas[:-1], sigmas[1:]):
            t = mx.array([s], dtype=mx.float32)
            v = tr(lat, t, cond if (base is None or s <= TEXT_IMAGES_MAX_NOISE) else base)
            if negative is not None:
                u = tr(lat, t, negative)
                v = u + a.guidance * (v - u)
            lat = lat + (s_next - s) * v
            mx.eval(lat)
        finals.append(lat)
        print(f"{tag}denoised in {time.perf_counter() - t0:.0f} s")
    del tr, negative, cond, base
    runs = [(out, prompt, caption, caption_seed, seed) for out, prompt, caption, caption_seed, seed, _, _ in runs]
    gc.collect()
    mx.clear_cache()

    # 3. the VAE decodes and saves every image, alone in memory
    vae = load_decoder(folder, dtype)
    for k, ((out, prompt, caption, caption_seed, seed), lat) in enumerate(zip(runs, finals), 1):
        tag = f"[{k}/{len(runs)}] " if len(runs) > 1 else ""
        t0 = time.perf_counter()
        image = vae.decode(lat)
        out = out or f"fibo2_edit_turbo_t2i_{seed}.png"
        image.save(out)
        record = {"model": a.model, "seed": seed, "steps": a.steps, "guidance_scale": a.guidance, "width": width,
                  "height": height, "prompt": prompt, "caption": caption, "precision": a.precision}
        if caption_seed is not None and caption:
            record["caption_seed"] = caption_seed
        with open(os.path.splitext(out)[0] + ".json", "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        print(f"{tag}saved {out} ({width}x{height}, {a.steps} steps, seed {seed}; decoded in "
              f"{time.perf_counter() - t0:.0f} s)")
    print(f"peak memory {mx.get_peak_memory() / 1e9:.1f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
