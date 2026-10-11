"""Per-job subprocess engine for Qwen-Image 2.1 (mflux/MLX).

Spawns one fresh, isolated interpreter per generation so the long-lived API worker's
process-global state can never corrupt denoising (fresh processes are provably clean).
Speaks JSON-lines on stdout: progress/phase updates and one final result dict.
"""
import gc
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

# Patch Qwen21WeightDefinition to handle single-file text_encoder layouts.
#
# Two community layouts ship one `text_encoder/model.safetensors` where the stock
# definition expects a sharded set plus an index:
#   * Sawfwair/Qwen-Image-2.1-Turbo-MLX-Mixed-4bit  -- single file, no index at all
#   * mlx-community/Qwen-Image-2.1-MLX-4bit        -- single file PLUS an index.json
#                                                      that names that same one file
# Both are `loading_mode="single"`. Everything else must fall through untouched so mflux
# raises its own FileNotFoundError naming the missing path.
#
# INVARIANT: this selector never returns None. The stock text_encoder definition has no
# variant_selector at all, so an `... if orig_selector else None` fallback returns None
# for every non-single layout, and mflux's weight_loader then dereferences that None --
# surfacing as the opaque "AttributeError: 'NoneType' object has no attribute
# 'download_url'". Returning the untouched base component instead lets mflux report the
# real problem.
try:
    from mflux.models.qwen21.weights.qwen21_weight_definition import Qwen21WeightDefinition
    from dataclasses import replace
    from pathlib import Path

    _QWEN_ORIG_GET_COMPONENTS = Qwen21WeightDefinition.get_components

    def _is_single_file_index(index_json_path: Path) -> bool:
        """True when index.json maps every tensor to the one model.safetensors."""
        try:
            weight_map = json.loads(index_json_path.read_text()).get("weight_map") or {}
            if not isinstance(weight_map, dict):
                return False
            files = set(weight_map.values())
            return len(files) == 1 and "model.safetensors" in files
        except Exception:
            return False

    def _patched_get_components(_orig=_QWEN_ORIG_GET_COMPONENTS):
        components = _orig()
        for c in components:
            if c.name != "text_encoder":
                continue

            def _te_variant_selector(root_path, _base=c, _orig_selector=c.variant_selector):
                te_path = root_path / _base.hf_subdir
                single_file = te_path / "model.safetensors"
                index_json = te_path / "model.safetensors.index.json"
                if single_file.is_file() and (
                    not index_json.is_file() or _is_single_file_index(index_json)
                ):
                    # replace(), not a hand-built ComponentDefinition: the latter silently
                    # drops precision / skip_quantization / key_transform / weight_files.
                    return replace(_base, loading_mode="single")
                return _orig_selector(root_path) if _orig_selector else _base

            c.variant_selector = _te_variant_selector
            break
        return components

    Qwen21WeightDefinition.get_components = staticmethod(_patched_get_components)
except Exception as _e:
    # stderr, not stdout: stdout is the JSON-lines transport and the parent discards
    # anything that is not valid JSON there.
    sys.stderr.write(f"[qwen_engine] Qwen21WeightDefinition patch failed: {_e}\n")

import gc
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import mlx.core as mx
from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21

MODEL = "mlx-community/Qwen-Image-2.1-MLX-4bit"
QWEN_SCHEDULERS = {"linear": "linear", "euler": "flow_match_euler_discrete"}
_MAX_PROMPT_BYTES = 128 * (1 << 10)

# (model_path, quantization) -> (pipeline, load_time_of_the_cold_load).
# The engine used to spawn one interpreter per generation, so this was rebuilt
# every single time: 2.1-2.9s of weight load measured on every Qwen job in the
# gallery history, never 0.0. The weights are a pure function of
# (model_path, quantization) - Qwen-Image 2.1 has no LoRA support at all - so a
# warm daemon can hand the same module to every subsequent request.
_PIPELINE_CACHE: dict = {}


def _get_pipeline(model_path, quantization, phase_cb=False):
    key = (str(model_path), int(quantization))
    hit = _PIPELINE_CACHE.get(key)
    if hit is not None:
        return hit[0], 0.0
    if phase_cb:
        emit({"phase": "loading_model", "detail": "Loading Qwen-Image-2.1 4-bit pipeline into unified memory..."})

    t0 = time.time()
    try:
        pipe = QwenImage21(quantize=quantization, model_path=str(model_path))
    except Exception as exc:
        # Name the model, the resolved path, and what is actually on disk. mflux's own
        # failure ("'NoneType' object has no attribute 'download_url'") says none of that,
        # which is why this cost a whole debugging session once already.
        te_dir = Path(model_path) / "text_encoder"
        try:
            present = ", ".join(sorted(p.name for p in te_dir.iterdir())) or "<empty>"
        except OSError:
            present = "<text_encoder/ unreadable>"
        raise RuntimeError(
            f"Qwen-Image 2.1 weight load failed: {type(exc).__name__}: {exc} | "
            f"model_path={model_path} quantize={quantization} | "
            f"text_encoder/ contains: {present}"
        ) from exc
    pipe.prompt_cache = _BoundedPromptCache(pipe.prompt_cache)
    load_time = round(time.time() - t0, 2)
    _PIPELINE_CACHE.clear()
    _PIPELINE_CACHE[key] = (pipe, load_time)
    return pipe, load_time


def _serve() -> int:
    """Read one JSON request per stdin line, write one JSON result per stdout line.

    Mirrors sdxl_engine's --serve mode and the parent's existing JSON-lines
    plumbing (_start_json_reader / _read_engine_message), so the parent can hold
    the process open between generations instead of paying a fresh interpreter
    plus a fresh weight load for each one.
    """
    def _handle_signal(signum, frame):
        raise KeyboardInterrupt

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(signum, _handle_signal)
        except (ValueError, OSError):
            pass

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            emit({"error": f"bad request: {e}"})
            continue
        if not isinstance(req, dict) or req.get("shutdown"):
            break
        try:
            _run(req, keep_alive=True)
        except KeyboardInterrupt:
            break
        except Exception as e:
            emit({"error": f"{type(e).__name__}: {e}"})
        gc.collect()
    _PIPELINE_CACHE.clear()
    try:
        mx.clear_cache()
    except Exception:
        pass
    return 0


def _wired_budget(requested=None):
    if requested is not None:
        try:
            return max(0, int(requested))
        except (TypeError, ValueError):
            pass
    try:
        d = mx.device_info()
        cap = d.get("max_recommended_working_set_size") or d.get("recommended_max_working_set_size") or 0
        mem = d.get("memory_size", 0)
        if mem > 0:
            budget = min(int(mem * 0.68), 9 * (1 << 30))
            if cap > 0:
                budget = min(budget, cap)
            return budget
    except Exception:
        pass
    return 9 * (1 << 30)


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _dump_env_if_requested():
    path = os.environ.get("QWEN_ENGINE_DUMP_ENV")
    if not path:
        return
    try:
        with open(path, "w") as f:
            f.write(json.dumps({k: str(v) for k, v in sorted(os.environ.items())}, indent=1))
    except Exception:
        pass


def _fingerprint(name, arr):
    try:
        import numpy as np
        mx.eval(arr)
        dt = None
        if getattr(arr, "dtype", None) is not None:
            dt = str(arr.dtype)
        if dt and dt.endswith("16"):
            arr = arr.astype(mx.float32)
        a = np.asarray(arr).reshape(-1)
        emit({"diag": name, "shape": list(getattr(arr, "shape", [])),
              "mean": float(a.mean()), "std": float(a.std()),
              "l1": float(np.abs(a).mean()),
              "min": float(a.min()), "max": float(a.max()),
              "first12": [float(x) for x in a[:12]]})
    except Exception as e:
        emit({"diag": name, "error": str(e)})


def _run_diagnostics(pipe, prompt, seed, height, width, steps, scheduler, eff_guidance):
    """QWEN_ENGINE_DIAG probe: fingerprint every intermediate the pipeline builds.

    Opt-in, and deliberately scoped so its tensors die with this call rather than
    staying resident across the render that follows.
    """
    from mflux.models.common.config.config import Config as _Config
    from mflux.models.common.config import ModelConfig as _MC
    from mflux.models.qwen21.latent_creator.qwen21_latent_creator import Qwen21LatentCreator as _LC
    from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_prompt_encoder import Qwen21PromptEncoder as _PE

    te = pipe.text_encoder
    tok = pipe.tokenizers["qwen21"]
    ids = mx.array(tok.tokenize(prompt).input_ids).reshape(1, -1)
    tok_emb = te.embed_tokens(ids.astype(mx.int32))
    mx.eval(tok_emb)
    _fingerprint("tok_emb_for_tokens", tok_emb)

    lat = _LC.create_noise(seed, height, width).astype(_MC.precision)
    mx.eval(lat)
    _fingerprint("noise", lat)

    emb, msk = _PE.encode_prompt(
        prompt=prompt,
        prompt_cache=pipe.prompt_cache,
        tokenizer=tok,
        text_encoder=te,
    )
    mx.eval(emb, msk)
    _fingerprint("text_embeds", emb)
    _fingerprint("text_mask", msk.astype(mx.float32))

    cfg = _Config(
        width=width, height=height, guidance=eff_guidance,
        scheduler=scheduler, image_path=None, image_strength=None,
        model_config=_MC.qwen_image_21(), num_inference_steps=steps,
    )
    t0 = next(iter(cfg.time_steps))
    out = pipe.transformer(
        t=t0, config=cfg,
        hidden_states=cfg.scheduler.scale_model_input(lat, t0),
        encoder_hidden_states=emb, encoder_hidden_states_mask=msk,
    )
    mx.eval(out)
    _fingerprint("transformer_first_step_out", out)
    del out, emb, msk, lat, tok_emb, ids
    mx.clear_cache()


def _value_bytes(value) -> int:
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes
    if isinstance(value, dict):
        return sum(_value_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_value_bytes(v) for v in value)
    return 0


class _BoundedPromptCache(dict):
    def __init__(self, initial=None):
        super().__init__(initial or {})
        self._trim()

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._trim()

    def _trim(self):
        total = sum(_value_bytes(v) for v in self.values())
        while self and (len(self) > 8 or total > 128 * (1 << 20)):
            key = next(iter(self))
            total -= _value_bytes(self.pop(key))


class _StepCB:
    def __init__(self, phase_cb=None):
        self._phase_cb = phase_cb

    def call_before_loop(self, **kwargs):
        if self._phase_cb:
            self._phase_cb("compiling", "Compiling Metal graph & encoding prompt...")

    def call_in_loop(self, t, **kwargs):
        emit({"progress": {"step": t + 1}})
        if self._phase_cb:
            self._phase_cb("generating", f"Denoising step {t + 1}...")

    def call_after_loop(self, **kwargs):
        if self._phase_cb:
            self._phase_cb("saving", "Decoding VAE latents & saving image...")


def main() -> int:
    _dump_env_if_requested()
    if "--serve" in sys.argv[1:]:
        return _serve()
    try:
        raw = sys.argv[1]
        if raw.lstrip().startswith("{"):
            req = json.loads(raw)
        else:
            req = json.loads(Path(raw).read_text())
    except Exception as e:
        emit({"error": f"bad request: {e}"})
        return 2
    return _run(req)


def _run(req: dict, keep_alive: bool = False) -> int:
    try:
        prompt = str(req.get("prompt") or "")
        if not prompt.strip():
            raise ValueError("prompt is required")
        if len(prompt.encode("utf-8")) > _MAX_PROMPT_BYTES:
            raise ValueError("prompt exceeds the 128 KiB limit")
        if not req.get("dest"):
            raise ValueError("dest is required")
        width = int(req["width"])
        height = int(req["height"])
        steps = int(req["steps"])
        if width < 128 or height < 128 or width > 2048 or height > 2048 or width % 16 or height % 16:
            raise ValueError("Qwen width and height must be multiples of 16 within 128-2048")
        if width * height > 589824:
            raise ValueError("Qwen-Image 2.1 is limited to 589824 pixels on 16GB Apple Silicon")
        if steps < 1 or steps > 25:
            raise ValueError("Qwen-Image 2.1 steps must be between 1 and 25 on this machine")
        negative = str(req.get("negative_prompt") or "")
        if len(negative.encode("utf-8")) > _MAX_PROMPT_BYTES:
            raise ValueError("negative prompt exceeds the 128 KiB limit")
        guidance = req.get("guidance")
        eff_guidance = 1.0 if guidance is None else float(guidance)
        if not math.isfinite(eff_guidance) or eff_guidance < 0:
            raise ValueError("guidance must be a finite non-negative number")
        if negative and eff_guidance <= 1.0:
            eff_guidance = 3.0
        sampler_label = str(req.get("sampler") or "linear").lower()
        if sampler_label not in QWEN_SCHEDULERS:
            raise ValueError(f"unsupported Qwen sampler: {sampler_label}")
        scheduler = QWEN_SCHEDULERS[sampler_label]
        seed = req.get("seed")
        if seed is None:
            seed = int(time.time())
        refs = [Path(p) for p in (req.get("ref_paths") or []) if p]
        if len(refs) > 1:
            raise ValueError("Qwen-Image 2.1 accepts at most one reference image")
        image_strength = req.get("image_strength")
        if image_strength is not None and not 0.0 < float(image_strength) <= 1.0:
            raise ValueError("image_strength must be in the interval (0, 1]")
        quantization = int(req.get("quantization", 4))
        if quantization != 4:
            raise ValueError("Qwen-Image 2.1 supports only 4-bit quantization")
    except (KeyError, TypeError, ValueError) as e:
        emit({"error": f"invalid request: {e}"})
        return 2

    set_limit = getattr(mx, "set_wired_limit", None) or mx.metal.set_wired_limit
    set_limit(_wired_budget(req.get("wired_limit_bytes")))

    pipe = None

    def phase(ph, detail=""):
        emit({"phase": ph, "detail": detail})

    def _handle_signal(signum, frame):
        raise KeyboardInterrupt

    old_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            old_handlers[signum] = signal.signal(signum, _handle_signal)
        except (ValueError, OSError):
            pass

    try:
        model_path = req.get("model_path") or MODEL
        pipe, load_time = _get_pipeline(model_path, quantization, phase_cb=bool(req.get("phase_cb")))

        needs_tiling = bool(refs) or max(width, height) > 512
        if needs_tiling:
            pipe.tiling_config = TilingConfig(
                vae_decode_tiles_per_dim=2,
                vae_decode_tile_size=512,
                vae_decode_overlap=8,
                vae_encode_tiled=bool(refs),
                vae_encode_tile_size=512,
                vae_encode_tile_overlap=64,
            )
        else:
            pipe.tiling_config = None

        # mx.compile() mutates the decoder in place, so the compiled version would
        # leak into the next request of a warm daemon. Keep the original and
        # restore it whenever this request did not ask for the compiled path.
        if not bool(req.get("fast_vae", True)) and hasattr(pipe, "vae") and hasattr(pipe.vae, "decoder"):
            try:
                raw_decoder = getattr(pipe, "_qwen_raw_decoder", None)
                if raw_decoder is None:
                    raw_decoder = pipe.vae.decoder
                    pipe._qwen_raw_decoder = raw_decoder
                pipe.vae.decoder = mx.compile(raw_decoder)
            except Exception:
                pass
        else:
            raw_decoder = getattr(pipe, "_qwen_raw_decoder", None)
            if raw_decoder is not None:
                pipe.vae.decoder = raw_decoder

        if os.environ.get("QWEN_ENGINE_DIAG"):
            # Its own function scope on purpose: this holds a full first-step transformer
            # output plus embeddings. Inline in _run they stay referenced until _run
            # returns, i.e. across the entire real generation -- on the 16 GB machine
            # that is the documented OOM. Returning frees them and clears the cache.
            _run_diagnostics(pipe, prompt, seed, height, width, steps, scheduler, eff_guidance)

        step_cb = _StepCB(phase_cb=(lambda ph, d: phase(ph, d)) if req.get("phase_cb") else None)
        pipe.callbacks.in_loop = [step_cb]
        pipe.callbacks.before_loop = [step_cb]
        pipe.callbacks.after_loop = [step_cb]

        t1 = time.time()
        out = pipe.generate_image(
            seed=seed,
            prompt=prompt,
            num_inference_steps=steps,
            height=height,
            width=width,
            guidance=eff_guidance,
            image_path=refs[0] if refs else None,
            image_strength=image_strength,
            negative_prompt=negative or None,
            scheduler=scheduler,
        )
        infer_time = round(time.time() - t1, 2)
        dest = Path(req["dest"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            out.image.save(dest)
        except Exception:
            try:
                dest.unlink()
            except OSError:
                pass
            raise
        emit({
            "generation_time": infer_time,
            "load_time": round(load_time, 2),
            "steps": steps,
            "sampler": sampler_label,
            "scheduler": scheduler,
            "guidance": eff_guidance,
            "width": width,
            "height": height,
            "quantization": quantization,
            "model_path": str(model_path),
            "tiled_vae": bool(pipe.tiling_config),
        })
        return 0
    except (Exception, KeyboardInterrupt) as e:
        try:
            import traceback
            traceback.print_exc(file=sys.stderr)
        except Exception:
            pass
        emit({"error": f"{type(e).__name__}: {e}"})
        return 1
    finally:
        for signum, handler in old_handlers.items():
            try:
                signal.signal(signum, handler)
            except (ValueError, OSError):
                pass
        if not keep_alive:
            # One-shot mode: give every byte back before the process exits.
            if pipe is not None:
                try:
                    pipe.prompt_cache.clear()
                except Exception:
                    pass
            pipe = None
            _PIPELINE_CACHE.clear()
            gc.collect()
            try:
                mx.clear_cache()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())