"""Per-job and daemon subprocess engine for OpenMOSS MOSS-Image (PyTorch / MPS).

Runs under the backend venv (PyTorch + MPS, transformers, diffusers).
Speaks JSON-lines on stdout: progress/phase updates and one final result dict.

Supports:
- MOSS-Image SFT 5B and Base 5B
- Diffusers MOSSImagePipeline with custom scheduler
- Spatial Box Layout JSON and plain-text prompts
- Apple Silicon MPS acceleration with CPU sinusoid fallback (preventing MPS float64 unsupported error)
- Dynamic memory offload: Qwen 3.5 text encoder executes on CPU, DiT denoises on MPS, FLUX.2 VAE decodes on MPS
- Reusable daemon mode (--serve) with parent-death watchdog
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import signal
import sys
import threading
import time
from pathlib import Path

import torch.nn.functional as F
from safetensors.torch import load_file

# Add backend directory to sys.path
_backend_dir = str(Path(__file__).resolve().parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

import torch
import torch.nn as nn
from PIL import Image

_stdout_lock = threading.Lock()
_pipe = None
_current_model_path = None

# Reused CPU embeddings keyed by (model_path, prompt_text). The empty-string
# negative is re-encoded on every generation otherwise (~50s on the 8.4B
# encoder); tensors are read-only consumers downstream (to()/repeat), so sharing
# them across requests inside the warm daemon is safe.
_text_embed_cache = {}


def _embed_cache_key(model_path: str, text: str) -> str:
    """SHA-256 over model path, the chat-template marker and the prompt text.

    The template bytes are pinned by name, not by content: if the tokenizer's
    chat template ever changes, the marker below must be bumped so stale
    embeddings on disk are not silently reused with the new template.
    """
    h = hashlib.sha256()
    h.update(model_path.encode("utf-8"))
    h.update(b"\x00template:qwen3_5-chat:enable_thinking=True\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


_EMBED_DISK_ENV = "MOSS_TE_EMBED_CACHE_DIR"


def _embed_disk_dir():
    """Directory of the persisted-embedding cache, or None.

    The Path is returned even when it does not exist yet: the load path
    probes the file (a missing directory just yields no hit) while the save
    path creates it. Gating on is_dir() would silently disable the cache
    forever on first use.
    """
    env = os.environ.get(_EMBED_DISK_ENV)
    if not env:
        return None
    return Path(env)


def _embed_disk_load(model_path: str, text: str):
    d = _embed_disk_dir()
    if d is None:
        return None
    import numpy as np
    p = d / f"{_embed_cache_key(model_path, text)}.npz"
    try:
        with np.load(p) as f:
            return np.asarray(f["embeds"]), np.asarray(f["mask"])
    except Exception:
        return None


def _embed_disk_save(model_path: str, text: str, embeds, mask) -> None:
    """Persist one (embeds, mask) pair as an .npz next to the settings dir.

    Failure to write is swallowed: the disk cache is an optimisation (embedding
    reuse across daemon restarts), never a correctness dependency.
    """
    d = _embed_disk_dir()
    if d is None:
        return
    import numpy as np
    try:
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{_embed_cache_key(model_path, text)}.npz"
        # numpy appends ".npz" to any name that does not already end in it, so
        # the temp file must keep that suffix or np.savez writes `*.tmp.npz`
        # and the replace then fails on a path that never existed.
        tmp = p.with_name(p.name + ".tmp.npz")
        np.savez_compressed(tmp, embeds=embeds, mask=mask)
        os.replace(tmp, p)
    except Exception:
        pass


# The MLX q8-g64 run of the same Qwen 3.5 encoder. Resolution order:
#   1. MOSS_TE_MLX set to an existing directory -> that directory.
#   2. MOSS_TE_MLX set to "0"/"off" -> torch CPU reference (escape hatch).
#   3. Unset -> auto-detect `<model dir>/../qwen3_5-te-q8g64`, i.e. the
#      converted encoder sitting next to the checkpoint in the model store;
#      absent, fall back to the torch CPU reference encoder. The converted
#      dir was produced once by moss_qwen35_mlx.convert_text_encoder and is
#      validated against the bf16 reference at cx 0.9995 (see its docstring).
#   4. Nothing local -> one-shot download of the pre-converted encoder from
#      HF (MOSS_TE_MLX_REPO, Ouinche/moss-image-qwen3.5-te-mlx-q8g64), so
#      release users never run the 19 GB bf16 + local-quantise path. A
#      failed download (offline, auth) silently falls back to torch CPU.
MOSS_TE_MLX_REPO_DEFAULT = "Ouinche/moss-image-qwen3.5-te-mlx-q8g64"
_TE_MLX_RESOLVE_CACHE: dict = {}


def _mlx_te_dir(model_path: str = "", download: bool = False):
    env = os.environ.get("MOSS_TE_MLX")
    if env:
        if env.strip().lower() in ("0", "off", "torch", "cpu"):
            return None
        p = Path(env)
        if p.is_dir():
            return p
    if model_path:
        sibling = Path(model_path).parent / "qwen3_5-te-q8g64"
        if sibling.is_dir():
            return sibling
    if not download:
        return None
    if model_path in _TE_MLX_RESOLVE_CACHE:
        return _TE_MLX_RESOLVE_CACHE[model_path]
    env = os.environ.get("MOSS_TE_MLX", "").strip()
    repo = None
    if env and env.lower() not in ("0", "off", "torch", "cpu") and not Path(env).is_dir() and "/" in env:
        repo = env  # env carries a HF repo id
    elif not env:
        repo = os.environ.get("MOSS_TE_MLX_REPO") or MOSS_TE_MLX_REPO_DEFAULT
    if not repo:
        return None
    try:
        from huggingface_hub import snapshot_download
        repo_dir = snapshot_download(
            repo_id=repo,
            allow_patterns=["*.json", "*.safetensors"],
        )
        print(f"TE_MLX_REPO_DOWNLOADED {repo} -> {repo_dir}", flush=True)
    except Exception as exc:
        print(f"TE_MLX_REPO_FAILED {repo}: {exc!r}", flush=True)
        return None
    _TE_MLX_RESOLVE_CACHE[model_path] = Path(repo_dir)
    return _TE_MLX_RESOLVE_CACHE[model_path]


def _load_tokenizer(model_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(Path(model_path) / "tokenizer"), local_files_only=True)


def _load_text_encoder(model_path: str):
    """Load the Qwen 3.5 text encoder + tokenizer (torch CPU reference path).

    Deliberately *not* cached: the encoder is ~16.8GB in bf16 and the
    transformer is ~10GB. On a 16GB Mac the two can never be resident at once,
    so the caller must free the encoder (see _drop_text_encoder) before the
    transformer is loaded. Only the produced embeddings are cached.

    If MOSS_TE_MLX points at a converted q8-g64 model directory, the encoder
    runs inside MLX instead (~10GB wired, dropped right after encoding);
    with nothing local the pre-converted encoder is pulled once from HF
    (MOSS_TE_MLX_REPO) so released installs never see the bf16 checkpoint.
    """
    from transformers import AutoModel

    mlx_dir = _mlx_te_dir(model_path, download=True)
    if mlx_dir is not None:
        return _load_mlx_text_encoder(mlx_dir, model_path), _load_tokenizer(model_path)

    path = Path(model_path)
    if not (path / "text_encoder").is_dir():
        raise RuntimeError(
            "MOSS text encoder unavailable: the MLX q8 encoder could not be "
            "resolved or downloaded, and this install has no bf16 "
            "text_encoder/ fallback. Check the network, or set MOSS_TE_MLX "
            "to a converted encoder directory.")
    tokenizer = _load_tokenizer(model_path)
    text_encoder = AutoModel.from_pretrained(
        str(path / "text_encoder"), dtype=torch.bfloat16, local_files_only=True).eval()
    return text_encoder, tokenizer


def _drop_text_encoder(text_encoder):
    """Free the encoder state before the 10GB transformer load.

    Order matters on a 16GB unified machine: mx.clear_cache() releases MLX's
    Metal buffer pool so the torch MPS allocator can claim the same physical
    pages. (`mx.clear_cache` -- not `mx.metal`, which is gone in current MLX.)
    """
    if isinstance(text_encoder, _MlxTextEncoder):
        text_encoder.model = None
        try:
            import mlx.core as mx
            before = mx.get_active_memory() + mx.get_cache_memory()
            gc.collect()
            mx.clear_cache()
            print("TE_MLX_DROPPED", json.dumps(
                {"before_gb": round(before / 1e9, 2),
                 "after_gb": round((mx.get_active_memory() + mx.get_cache_memory()) / 1e9, 2)}),
                flush=True)
        except Exception as exc:
            print("TE_MLX_CLEAR_FAILED", repr(exc), flush=True)
    del text_encoder
    gc.collect()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


class _MlxTextEncoder:
    """Drop-in torch interface for the mlx-lm Qwen 3.5 4bit text encoder.

    Accepts the transformer-style `inputs` kwargs the torch pipeline uses
    (input_ids / attention_mask as batch=1 tensors) and returns an object with
    `last_hidden_state` shaped [1, seq, hidden], identical contract to the
    torch encoder path. Weights stay in MLX; output leaves as a torch
    CPU tensor so nothing in the DiT path changes.
    """

    def __init__(self, model, tokenizer=None, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, *, input_ids=None, attention_mask=None, use_cache=False,
                 return_dict=True, **_ignored):
        import mlx.core as mx
        import numpy as np

        ids = mx.array(input_ids.to("cpu").numpy())
        # TextModel.__call__ runs the lm_head; the encoder needs only the
        # post-final-norm hidden states, so drive Qwen3_5TextModel directly.
        # GDN expects [B, S, H]; row 0 is the unpadded sequence (batch=1).
        # MLX bf16 has no numpy buffer protocol (PEP 3118), so widen in MLX
        # first; the torch side then narrows back to bf16. Fused before eval
        # to eliminate redundant GPU-CPU synchronization stalls.
        hidden = self.model.model(ids)[0].astype(mx.float32)
        mx.eval(hidden)
        # Conversion AFTER eval: np.array on an un-evaluated MLX array aborts
        # the whole process (Metal failure inside numpy's C callback), so this
        # is deliberately the evaluated form.
        nprow = np.asarray(hidden, dtype=np.float32)
        hidden_t = torch.from_numpy(nprow).to(self.device, dtype=torch.float32)
        return _TextOutput(
            last_hidden_state=hidden_t.unsqueeze(0).to(torch.bfloat16))


class _TextOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


def _load_mlx_text_encoder(mlx_dir: Path, model_path: str):
    """Construct the _MlxTextEncoder wrapper via the shared local converter.

    mlx_lm.load is deliberately NOT used here: it reads `quantization` from
    config.json and quantises every Linear/Embedding, including the GDN
    projections our converter deliberately left in bf16 -- the load would then
    silently diverge from the torch reference. moss_qwen35_mlx applies the
    same predicate before load_weights, so the runtime model shape matches
    the converted file exactly.
    """
    from moss_qwen35_mlx import load_mlx_text_model

    model = load_mlx_text_model(mlx_dir)
    tokenizer = _load_tokenizer(model_path)
    return _MlxTextEncoder(model=model, tokenizer=tokenizer)


def _encode_prompt(text_encoder, tokenizer, model_path: str, text: str,
                   max_sequence_length: int = 4096):
    """Encode one prompt, with in-memory and persisted-disk caches.

    The chat template (`enable_thinking=True`), the truncation call and the
    per-row unpadded forward contract mirror `pipeline_moss_image.encode_prompt`
    exactly -- both encoder backends (torch CPU and the MLX q4 shim) consume
    the same tokens, so the cache is valid regardless of which backend first
    produced a given prompt's embedding.
    """
    cache_key = (model_path, text)
    cached = _text_embed_cache.get(cache_key)
    if cached is not None:
        return cached
    disk = _embed_disk_load(model_path, text)
    if disk is not None:
        import numpy as np
        embeds = torch.from_numpy(np.asarray(disk[0]))
        mask = torch.from_numpy(np.asarray(disk[1]))
        if embeds.dtype != torch.bfloat16:
            embeds = embeds.to(torch.bfloat16)
        _text_embed_cache[cache_key] = (embeds, mask)
        return embeds, mask
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True, enable_thinking=True)
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True,
                       max_length=max_sequence_length)
    with torch.no_grad():
        if isinstance(text_encoder, _MlxTextEncoder):
            output = text_encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                use_cache=False, return_dict=True)
        else:
            output = text_encoder(**inputs, use_cache=False, return_dict=True)
    row = output.last_hidden_state[0]
    embeds = row.unsqueeze(0).to("cpu", torch.bfloat16)
    mask = torch.ones(embeds.shape[:2], dtype=torch.bool)
    if len(_text_embed_cache) >= 8:
        _text_embed_cache.clear()
    _embed_disk_save(model_path, text,
                     embeds.to(torch.float32).numpy(),
                     mask.numpy())
    _text_embed_cache[cache_key] = (embeds, mask)
    return embeds, mask


def emit(payload: dict) -> None:
    """Emit a JSON-lines message to stdout for the parent process."""
    line = json.dumps(payload, separators=(",", ":")) + "\n"
    with _stdout_lock:
        sys.stdout.write(line)
        sys.stdout.flush()


def _patch_moss_transformer(transformer_cls) -> None:
    """Patch MOSSImageTransformer.forward to make time embeddings Apple Silicon MPS safe.

    Apple Silicon MPS does not support torch.float64. MOSSImageTransformer's original
    time embedding calculates sinusoidal frequencies with torch.float64 on the input
    device (times.device), which raises:
      TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64.
    Computing the 128 frequencies on CPU in float64 and moving the resulting float32
    embedding back to the execution device completely resolves this without any loss of precision.
    """
    if getattr(transformer_cls, "_mps_patched", False):
        return

    def mps_safe_forward(self, hidden_states, timestep, encoder_hidden_states,
                         encoder_attention_mask=None, return_dict=True, skip_layers=None):
        from pipeline_moss_image import _validate_skip_layers, Transformer2DModelOutput
        skip_layers = _validate_skip_layers(skip_layers, len(self.layers))
        if hidden_states.ndim != 4 or hidden_states.shape[1] != self.config.latent_channels:
            raise ValueError("hidden_states must be [B, latent_channels, H, W].")
        batch, _, height, width = hidden_states.shape
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != batch:
            raise ValueError("encoder_hidden_states must be [B, S, text_dim].")
        mask = encoder_attention_mask
        if mask is None:
            mask = torch.ones(encoder_hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        if mask.shape != encoder_hidden_states.shape[:2] or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("encoder_attention_mask must be a binary [B, S] mask.")
        times = torch.as_tensor(timestep, device=hidden_states.device, dtype=torch.float32).reshape(-1)
        times = times.expand(batch)

        # MPS-safe sinusoidal frequency computation on CPU in float64
        half = self.config.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device="cpu", dtype=torch.float64) / half)
        phase = times.cpu().double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float().to(times.device)

        time_emb = self.time_embedding(embedding.to(self.time_embedding[0].weight.dtype)).float()
        mod = self.shared_block_modulation(time_emb.to(self.shared_block_modulation.weight.dtype)).view(batch, 6, -1)
        head = self.video_head_modulation(time_emb.to(self.video_head_modulation.weight.dtype)).view(batch, 2, -1)
        outputs = []
        for index in range(batch):
            context = encoder_hidden_states[index:index + 1, mask[index].bool()]
            if context.shape[1] == 0:
                raise ValueError("Each sample needs at least one valid text token.")
            video = hidden_states[index:index + 1].flatten(2).transpose(1, 2)
            video = self.video_embedder(video.to(self.video_embedder.weight.dtype).contiguous())
            rope = self.rope(height, width, context.shape[1], hidden_states.device).to(video.dtype)
            text = self.text_embedder(context, rope[:, height * width:])
            x = torch.cat((video, text), 1)
            for layer_index, layer in enumerate(self.layers):
                if layer_index not in skip_layers:
                    x = layer(x, rope, mod[index:index + 1])
            x = self.final_norm_video(x[:, :height * width].to(self.final_norm_video.weight.dtype))
            x = x * (1 + head[index:index + 1, 1, None]) + head[index:index + 1, 0, None]
            x = self.final_linear_video(x.to(self.final_linear_video.weight.dtype))
            outputs.append(x.transpose(1, 2).reshape(1, -1, height, width))
        sample = torch.cat(outputs)
        return Transformer2DModelOutput(sample=sample) if return_dict else (sample,)

    transformer_cls.forward = mps_safe_forward
    transformer_cls._mps_patched = True


class _LoRALinear(nn.Module):
    """Frozen Linear plus one kohya LoRA delta.

    y = W x + (alpha / rank) * up(down(x)) * scale

    The adapter keeps its stored dtype (fp32 for the DMD adapter); only the
    resulting delta is cast to the output dtype, so a bf16 transformer is never
    silently promoted and the low-precision delta is never rounded away before
    it is added.
    """

    def __init__(self, base, down, up, scale):
        super().__init__()
        self.base = base
        self.down = nn.Parameter(down, requires_grad=False)
        self.up = nn.Parameter(up, requires_grad=False)
        self.scale = float(scale)

    @property
    def weight(self):
        # The reference DiT blocks read proj['q'].weight.dtype before their
        # attention projections; the wrapper must expose the base Linear's
        # canonical tensors or every wrapped target breaks the forward pass.
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        out = self.base(x)
        delta = F.linear(F.linear(x.to(self.down.dtype), self.down), self.up) * self.scale
        return out + delta.to(out.dtype)


def _resolve_module(root, dotted):
    """Resolve a dotted path through Attributes, ModuleList and ModuleDict."""
    obj = root
    for part in dotted.split("."):
        if isinstance(obj, nn.ModuleDict):
            if part not in obj:
                return None
            obj = obj[part]
        elif part.isdigit() and isinstance(obj, (nn.ModuleList, nn.Sequential, list, tuple)):
            idx = int(part)
            if idx >= len(obj):
                return None
            obj = obj[idx]
        elif hasattr(obj, part):
            obj = getattr(obj, part)
        else:
            return None
    return obj


def _set_module(root, dotted, new):
    parts = dotted.split(".")
    parent = _resolve_module(root, ".".join(parts[:-1]))
    if parent is None:
        return False
    last = parts[-1]
    if isinstance(parent, nn.ModuleDict):
        parent[last] = new
    else:
        setattr(parent, last, new)
    return True


_LORA_KEY_RE = re.compile(
    r"^(?:(?:diffusion_model|transformer)\.)?(.+?)\.(lora_down|lora_up|lora_A|lora_B)(?:\.weight)?$")
_LORA_ALPHA_RE = re.compile(r"^(?:(?:diffusion_model|transformer)\.)?(.+?)\.alpha$")


def _group_lora_state_dict(state):
    """Collect {module_path: {down, up, alpha}} from a kohya/diffusers adapter."""
    grouped = {}
    for key, value in state.items():
        alpha_match = _LORA_ALPHA_RE.match(key)
        if alpha_match:
            grouped.setdefault(alpha_match.group(1), {})["alpha"] = value
            continue
        match = _LORA_KEY_RE.match(key)
        if not match:
            continue
        base, kind = match.group(1), match.group(2)
        slot = grouped.setdefault(base, {})
        slot["down" if kind in ("lora_down", "lora_A") else "up"] = value
    return grouped


def _apply_loras(transformer, loras):
    """Apply kohya LoRA adapters onto the MOSS transformer's projection linears.

    A target key that cannot be resolved raises rather than being skipped: a
    silently ignored adapter is the failure this whole path exists to prevent.
    """
    applied = 0
    for entry in loras or []:
        path = entry.get("path")
        if not path:
            continue
        scale = float(entry.get("scale", 1.0) or 1.0)
        state = load_file(str(path))
        grouped = _group_lora_state_dict(state)
        for base, slot in grouped.items():
            down, up = slot.get("down"), slot.get("up")
            if down is None or up is None:
                continue
            rank = int(down.shape[0])
            alpha = slot.get("alpha")
            alpha_value = float(alpha.reshape(-1)[0]) if alpha is not None else float(rank)
            eff_scale = scale * (alpha_value / rank)
            target = _resolve_module(transformer, base)
            if target is None or not isinstance(target, nn.Linear):
                raise RuntimeError(f"LoRA target {base} is unavailable")
            wrapped = _LoRALinear(target, down.float(), up.float(), eff_scale)
            if not _set_module(transformer, base, wrapped):
                raise RuntimeError(f"could not attach LoRA at {base}")
            applied += 1
    return applied


def _unwrap_loras(transformer):
    """Restore every wrapped linear to its base module, in place.

    _apply_loras mutates the *cached* pipeline, so a second request would
    re-apply onto _LoRALinear targets (which are not nn.Linear) and the strict
    availability check would raise. Unwrapping first also purges the previous
    run's delta when the next request changes or drops its LoRA set: a wrapper
    never gets merged into the base weights, so leaving one attached carries
    the old adapter into a generation that never asked for it.
    """
    restored = 0
    for path, module in list(transformer.named_modules()):
        if isinstance(module, _LoRALinear):
            if _set_module(transformer, path, module.base):
                restored += 1
    return restored


def _drop_pipeline():
    """Release the cached pipeline before a model switch.

    The old code only reassigned `_pipe`; the previous transformer, VAE and
    text encoder stayed alive until Python's next GC cycle decided otherwise,
    so an SFT->Base hop held two full 5B pipelines in unified memory at once.
    """
    global _pipe, _current_model_path
    if _pipe is None:
        return
    _pipe = None
    _current_model_path = None
    _text_embed_cache.clear()
    gc.collect()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _get_pipeline(model_path: str, adapter: str = ""):
    """Load and cache the *denoising* half of the pipeline (transformer + VAE).

    The text encoder is loaded and freed separately (see _load_text_encoder):
    loading the whole pipeline with from_pretrained keeps the 16.8GB encoder and
    the 10GB transformer resident at the same time, which does not fit on a 16GB
    machine. The returned pipeline therefore has text_encoder=tokenizer=None and
    is always driven with pre-computed embeddings.

    ``adapter`` (a distilled-adapter directory, "" for the stock model) is part
    of the cache key: the DMD replacements (fp32 time embedder, wrapped LoRA
    linears) permanently mutate the cached transformer, so a stock request must
    not silently reuse a distilled pipeline and vice versa.
    """
    global _pipe, _current_model_path
    cache_key = f"{model_path}|{adapter}"
    if _pipe is not None and _current_model_path == cache_key:
        return _pipe, 0.0
    if _pipe is not None:
        _drop_pipeline()

    emit({"phase": "loading_model", "detail": "Loading MOSS-Image transformer + VAE..."})
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    # Ensure model folder is in sys.path so pipeline_moss_image can resolve components
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

    from diffusers import AutoencoderKLFlux2

    from pipeline_moss_image import MOSSImagePipeline, MOSSImageScheduler, MOSSImageTransformer
    _patch_moss_transformer(MOSSImageTransformer)

    t0 = time.time()
    transformer = MOSSImageTransformer.from_pretrained(
        str(path / "transformer"), torch_dtype=torch.bfloat16, local_files_only=True).eval()
    _patch_moss_transformer(transformer.__class__)
    # The bundled decoder is the FLUX.2 small VAE and runs in float32 (the model
    # card's {"default": bf16, "vae": fp32}); bf16 decoding loses detail.
    vae = AutoencoderKLFlux2.from_pretrained(
        str(path / "vae"), torch_dtype=torch.float32, local_files_only=True).eval()
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    if hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
    # MOSSImageScheduler carries the training shift (5.0) in its config; building
    # it fresh is exactly the card's "re-instantiate the scheduler" step.
    pipe = MOSSImagePipeline(transformer, MOSSImageScheduler(), None, None, vae)

    _pipe = pipe
    _current_model_path = cache_key
    load_time = round(time.time() - t0, 2)
    return _pipe, load_time


def _apply_distilled_time_modules(transformer, adapter_dir: str) -> None:
    """Replace the step embedder and modulation biases with the DMD fp32 tensors.

    The distilled checkpoint folds the constant t=1 time vector (encoded as
    step codes that no longer match the 1000-step schedule) into the biases of
    the shared modulation and video-head modulation branches, so the stock
    time embedding must be swapped as a whole together with those biases. They
    are fp32 in the checkpoint file: the distillation delta is smaller than one
    bf16 rounding step, so keeping them bf16 would erase it.
    """
    if getattr(transformer, "_distilled_adapter_dir", None) == adapter_dir:
        return
    for name in ("time_embedding", "shared_block_modulation", "video_head_modulation"):
        getattr(transformer, name).float()
    state = load_file(str(Path(adapter_dir) / "time_embedding.safetensors"), device="cpu")
    result = transformer.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(
            f"distilled time_embedding.safetensors has unexpected keys: {result.unexpected_keys[:5]}")
    transformer._distilled_adapter_dir = adapter_dir


def _distilled_decode(pipe, x0):
    """Decode a dataspace-sampler clean latent with the release VAE.

    The DMD samplers iterate in the DiT's packed latent space; the release
    pipeline's normal decode path is bypassed entirely, so the BatchNorm
    denormalisation and the 2x2 unpatchify that the pipeline performs inside
    its own decode have to be replicated here (protocol from the adapters'
    reference inference.py).
    """
    vae = pipe.vae
    lat = x0.to(next(vae.parameters()).dtype)
    bn = vae.bn
    eps = float(getattr(vae.config, "batch_norm_eps", 1e-5))
    lat = lat * (bn.running_var + eps).sqrt().view(1, -1, 1, 1) + bn.running_mean.view(1, -1, 1, 1)
    b, c, h, w = lat.shape
    lat = lat.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)
    image = vae.decode(lat, return_dict=False)[0].clamp(-1, 1)
    image = pipe.image_processor.postprocess(image, output_type="pil")
    return image[0]


def _run(req: dict, keep_alive: bool = False) -> int:
    try:
        prompt = str(req.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("prompt is required")
        dest = req.get("dest")
        if not dest:
            raise ValueError("dest is required")
        model_path = req.get("model_path")
        if not model_path:
            raise ValueError("model_path is required")

        width = int(req.get("width", 1024))
        height = int(req.get("height", 1024))
        if width % 16 or height % 16 or width < 128 or height < 128:
            raise ValueError(f"Dimensions must be multiples of 16 >= 128 (got {width}x{height})")

        steps = int(req.get("steps", 50))
        if steps < 1:
            raise ValueError("steps must be >= 1")

        guidance = float(req.get("guidance", 3.0))
        seed = req.get("seed")
        if seed is None:
            seed = int(time.time())
        negative_prompt = str(req.get("negative_prompt") or "").strip()

        adapter_dir = str(req.get("adapter_dir") or "").strip()
        # custom:<name> requests a user-declared sampler from the samplers
        # store; it only ever applies to the stock pipeline (DMD adapters own
        # their schedule). A broken or undeclared class falls back silently.
        custom_sampler_id = str(req.get("sampler") or "").strip()
        if adapter_dir or not custom_sampler_id.startswith("custom:"):
            custom_sampler_id = None
        if adapter_dir and not Path(adapter_dir).is_dir():
            raise FileNotFoundError(f"distilled adapter dir does not exist: {adapter_dir}")

        # Stage 1 -- text encoder alone on the CPU. It is freed before the
        # transformer is loaded, because the two do not fit together on 16GB.
        # Check caches first: skip loading the 4.5 GB weights if embeddings are cached.
        do_cfg = guidance != 1.0 and not adapter_dir
        negative_embeds, negative_mask = None, None

        has_prompt_cache = (model_path, prompt) in _text_embed_cache or _embed_disk_load(model_path, prompt) is not None
        has_neg_cache = (not do_cfg) or (model_path, negative_prompt) in _text_embed_cache or _embed_disk_load(model_path, negative_prompt) is not None

        if has_prompt_cache and has_neg_cache:
            emit({"phase": "encoding_prompt", "detail": "Reusing cached Qwen 3.5 prompt embeddings..."})
            text_encoder, tokenizer = None, None
        else:
            mlx_dir = _mlx_te_dir(model_path, download=True)
            if mlx_dir is not None:
                emit({"phase": "encoding_prompt", "detail": "Encoding prompt with Qwen 3.5 (MLX q8g64, GPU)..."})
            else:
                emit({"phase": "encoding_prompt", "detail": "Encoding prompt with Qwen 3.5 (CPU)..."})
            t_start = time.time()
            text_encoder, tokenizer = _load_text_encoder(model_path)

        # Every exit path must drop the encoder if loaded: a failed encode otherwise leaves
        # ~10 GB of MLX Metal buffers cached in the warm daemon, which is exactly
        # the headroom the transformer load needs next.
        try:
            prompt_embeds, prompt_attention_mask = _encode_prompt(
                text_encoder, tokenizer, model_path, prompt)
            if do_cfg:
                # Pre-encode the negative (empty string included): pipe() would
                # otherwise re-encode it internally on the execution device.
                negative_embeds, negative_mask = _encode_prompt(
                    text_encoder, tokenizer, model_path, negative_prompt)
        finally:
            if text_encoder is not None:
                _drop_text_encoder(text_encoder)
                text_encoder = None

        # Stage 2 -- transformer + VAE on MPS.
        pipe, load_time = _get_pipeline(model_path, adapter=adapter_dir)

        # LoRA wrappers live on the cached transformer: restore the base
        # linears before this run's set is applied (or none is), or a repeat
        # request fails the strict target check and a dropped LoRA leaks its
        # delta into the next generation.
        unwrapped = _unwrap_loras(pipe.transformer)
        if unwrapped:
            emit({"diag": f"moss: unwrapped {unwrapped} LoRA linears from the cached pipeline"})

        loras = req.get("loras") or []
        if adapter_dir:
            # The distilled lora.safetensors is kohya-format, so the regular
            # wrapper applies it unmerged at 1.0 (the card's recommended
            # strength); the time embedder / modulation biases come separately.
            _apply_distilled_time_modules(pipe.transformer, adapter_dir)
            if not loras:
                loras = [{"path": str(Path(adapter_dir) / "lora.safetensors"), "scale": 1.0}]
        if loras:
            emit({"phase": "loading_loras", "detail": f"Applying {len(loras)} LoRA(s) to the transformer..."})
            applied = _apply_loras(pipe.transformer, loras)
            emit({"phase": "loading_loras", "detail": f"Applied {applied} LoRA projection(s)."})

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        pipe.transformer.to(device)
        pipe.vae.to(device)

        # Step 3: Setup step progress callback
        def _make_progress_bar(total):
            def _bar(iterable):
                for idx, val in enumerate(iterable):
                    emit({"progress": {"step": idx + 1, "steps": total}})
                    yield val
            return _bar

        pipe.progress_bar = _make_progress_bar(steps)
        generator = torch.Generator(device="cpu").manual_seed(seed)

        if adapter_dir:
            sampler_label = "moss_dataspace_dmd"
        else:
            sampler_label = "flow_match_euler_discrete"

        emit({"phase": "denoising", "detail": f"Denoising {steps} steps on Apple Silicon..."})
        with torch.no_grad():
            if adapter_dir:
                ds = load_file(str(Path(adapter_dir) / "dataspace.safetensors"), device=str(device))
                Q = ds["latent_mix.Q"].to(device=device, dtype=torch.float32)
                codes = ds["step_codes"].to(device=device, dtype=torch.float32)
                steps = min(steps, int(codes.shape[0]))
                z = torch.randn(
                    (1, int(pipe.transformer.config.latent_channels),
                     height // 16, width // 16),
                    generator=generator, dtype=torch.float32).to(device)
                x0 = torch.zeros_like(z)
                emb = prompt_embeds.to(device)
                msk = prompt_attention_mask.to(device)
                for r in range(steps):
                    emit({"progress": {"step": r + 1, "steps": steps}})
                    x = z + torch.einsum("oc,bchw->bohw", Q, x0)
                    x0 = (z - pipe.transformer(x, codes[r], emb, msk).sample.float())
                image = _distilled_decode(pipe, x0)
            else:
                if custom_sampler_id:
                    try:
                        import samplers_store
                        cls = samplers_store.get_class(custom_sampler_id.split(":", 1)[1], "moss")
                        if cls is None:
                            raise ValueError("not declared for the moss engine or broken")
                        pipe.scheduler = cls()
                        sampler_label = custom_sampler_id
                    except Exception as exc:
                        emit({"diag": f"custom sampler {custom_sampler_id} unavailable: {exc}"})
                        custom_sampler_id = None
                output = pipe(
                    prompt_embeds=prompt_embeds,
                    prompt_attention_mask=prompt_attention_mask,
                    negative_prompt_embeds=negative_embeds,
                    negative_prompt_attention_mask=negative_mask,
                    negative_prompt="" if negative_embeds is not None else negative_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=generator,
                    uncond_skip_layers=(22,) if do_cfg else None,
                )
                image = output.images[0]

        emit({"phase": "saving", "detail": "Finalizing and saving image..."})
        dest_path = Path(dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        def _diag(label: str):
            # Stage-by-stage trace of the only two seconds that have ever lost
            # an image: save, replace, verify, emit. One tiny line each into
            # a sidecar next to the raw; deleted together with the part file
            # after a successful replace so a clean run leaves no litter.
            try:
                with open(dest_path.with_name(f".{dest_path.name}.diaglog"),
                          "a", encoding="utf-8") as fh:
                    fh.write(f"{time.time():.3f} {_diag.request} {label} pid={os.getpid()}\n")
            except OSError:
                pass

        _attr = f"req:{dest_path.name[:6]}"
        if not hasattr(_diag, "id"):
            _diag.counter = 0
        _diag.counter += 1
        _diag.request = f"r{_diag.counter}{_attr}"
        # Atomic save: PIL writes the target in place, so a half-written file on a
        # slow external volume is visible to the parent mid-save. The parent opens
        # the raw the instant it accepts the success message; hand it a file that
        # is either fully there or not there at all. Explicit format because the
        # `.part<pid>` suffix is not a known image extension -- PIL infers the
        # encoder from the file name and would raise "unknown file extension".
        _diag("save-start")
        part_path = dest_path.with_name(f".{dest_path.name}.part{os.getpid()}")
        image.save(part_path, format="PNG")
        _diag("save-done")
        os.replace(part_path, dest_path)
        _diag("replace-done")
        # Flush both the file and the directory entry: the parent may open this
        # path within milliseconds across the JSON pipe, and a directory entry
        # that exists logically but was never fsynced is one reboot away from
        # "missing".
        try:
            _dir_fd = os.open(dest_path.parent, os.O_RDONLY)
            try:
                os.fsync(_dir_fd)
            finally:
                os.close(_dir_fd)
        except OSError:
            pass

        # Refusing to claim success for a file that did not survive the replace:
        # the parent crashes with a bare [Errno 2] at Image.open, which looks
        # nothing like a save problem and everything like a code bug.
        try:
            dest_size = os.path.getsize(dest_path)
        except OSError:
            dest_size = 0
        if dest_size <= 0:
            raise IOError(
                f"image save did not persist: {dest_path} is "
                f"{'missing' if dest_size == 0 and not os.path.exists(dest_path) else 'empty'} "
                f"after an os.replace that reported success")

        gen_time = round(time.time() - t_start, 2)
        _diag("verified")
        emit({
            "status": "success",
            "generation_time": gen_time,
            "load_time": load_time,
            "dest_size": dest_size,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance": guidance,
            "seed": seed,
            "sampler": sampler_label,
            "distilled": bool(adapter_dir),
        })
        _diag("success-emit")
        try:
            dest_path.with_name(f".{dest_path.name}.diaglog").unlink()
        except OSError:
            pass
        return 0
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        emit({"error": str(e)})
        return 1
    finally:
        # Warm-daemon hygiene: without this the allocator's MPS cache and the
        # last generation's tensor graph stay resident between requests, so a
        # sequence of jobs grows until the wired budget is exhausted.
        if keep_alive:
            gc.collect()
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()


def _serve() -> int:
    """Daemon mode: read JSON requests line by line from stdin."""
    # Arm parent-death watchdog
    try:
        from engine_watchdog import arm_orphan_exit, parent_pid_from_argv
        _expected_parent = parent_pid_from_argv(sys.argv[1:])
        if _expected_parent is not None:
            arm_orphan_exit(_expected_parent)
    except Exception:
        pass

    emit({"status": "ready", "engine": "moss"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            emit({"error": f"invalid json: {e}"})
            continue

        if req.get("command") == "ping":
            emit({"status": "pong"})
            continue
        if req.get("command") == "shutdown":
            break

        _run(req, keep_alive=True)

    return 0


def main() -> int:
    def _sig_handler(signum, frame):
        sys.exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _sig_handler)
        except (ValueError, OSError):
            pass

    if "--serve" in sys.argv:
        return _serve()

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg.startswith("{"):
            req = json.loads(arg)
        else:
            req = json.loads(Path(arg).read_text(encoding="utf-8"))
        return _run(req)

    sys.stderr.write("Usage: moss_engine.py [--serve] [<request_json>]\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
