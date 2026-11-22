# Vendored mflux — provenance & local patches

- **Upstream:** https://github.com/mflux-community/mflux (fork of filipstrand/mflux)
- **Base commit:** `014252f` on `main` (2026-11-18, "Qwen-Image-2.1: --enhance-prompt rewrites t2i too (#998)"); last release tag at that point: `v.0.24.0` (PyPI 0.24.0, 2026-10-27). Previous vendored base: `v.0.23.0` (63cec2b).
- **Rebase date:** 2026-11-19. Method: 3-way merge per file (`git merge-file`, base `v.0.23.0`, ours = previous vendored tree, theirs = `main`), with conflicts resolved by hand toward our patch semantics (see below). `models/flux3/` is kept wholesale from our tree: the backend (`flux3_spatial.py`, Virtual Boxes) is wired to our implementation.
- **Packaging:** flat layout (`mflux-src/mflux/` + setuptools pyproject), installed with `pip install --no-deps` (see `packaging/build_runtime_venv.sh`).

## What the rebase brings (upstream `v.0.23.0` → `main`)

New model ports: **DreamLite / DreamLite-mobile** (#997), **Anima Turbo/Aesthetic/Base** (#931 + training #943), **Ideogram 4 FP8** (#20/#2427 area), **ERNIE Image**, **SenseNova-U1.5**, **Iris-3B**, Boogu, Ming-Image, HED/OpenPose/DepthPro/SeedVR2 (control/edit), Qwen-Image-2.1 **ControlNet + LoRA** (#859), FLUX.3 LoRA loading fixes (#966–#968), small-text-encoder options for FLUX.2/Z-Image/Qwen-Image-2.1 (#974/#980/#941), LoRA training adapters (`mflux-train`) for Qwen-Image-2.1/Anima, `--enhance-prompt` on t2i (#998), mflux-save shard-aware download patterns (#869), better missing-weight diagnostics.

## DiffusionBear patches carried over (do not drop on the next rebase)

1. **Process-safety — evaluate before `np.array()`** (Metal failure inside numpy's C buffer callback → `std::terminate` instead of a catchable error): `mflux/utils/image_util.py` (`mx.eval(images)` before `np.array`) and `mflux/models/common/vae/vae_tiler.py` (per-tile `mx.eval` + `mx.clear_cache()`). Guarded by `test_vae_tiler_eval_first.py` / `test_vendor_image_util_eval_first.py`.
2. **PyTorch-free loaders**: `models/common/pid_decoder/pid_weight_mapping.py` (safetensors-only `convert_checkpoint`), `models/common/weights/loading/weight_loader.py` (`mx.load` instead of `torch.load`/`safetensors.torch`, bf16→fp16 via `astype`), `models/qwen/tokenizer/qwen_vision_language_processor.py` and `models/fibo_vlm/tokenizer/qwen2vl_processor.py` (`return_tensors="np"` only). `torch` is not a dependency of this vendored build.
3. **16 GB text-encoder strategy**: `krea2/weights/krea2_weight_definition.py` (TE quantized q4 group_size=32, `skip_quantization=False`, root-level shard patterns), `qwen/weights/qwen_weight_definition.py` and `qwen21/weights/qwen21_weight_definition.py` (`skip_quantization=False`, no forced bf16 TE; the backend persists its own `<model>-te-q4-g64-<sha>.safetensors` cache and drops `text_encoder` from the models dict in `WeightApplier._set_weights` before pipeline construction).
4. **Quantization detection fallback** in `weight_loader.py`: root `config.json` `quantization.bits` / `quantization_level`, then `.scales` key sniffing.
5. **M1/M2 warmup-then-compile** (shape-specialised, NEVER `shapeless=True` — `Split` output-shape inference dies at denoise step 3): `krea2/variants/txt2img/krea2.py`, `z_image/variants/z_image.py`, `flux2/variants/txt2img/flux2_klein.py`, `flux2/variants/edit/flux2_klein_edit.py` (always `mx.compile`, never shapeless).
6. **FLUX.2 KV-cache reference path** in `flux2_klein.py` / `flux2_klein_edit.py` (`use_kv_cache`, `_create_kv_caches`, `_cached_predict`) — used by the app's img2img/fill; upstream moved this off the txt2img variant.
7. **LoRA/weight mappings**: `flux2/weights/flux2_weight_mapping.py` and `qwen21/weights/qwen21_weight_mapping.py` extended for quantized `.scales`/`.biases` leaves, dual HF/community key prefixes, conv2d layout auto-detect; `weight_transforms.squeeze_to_2d` (used by flux3 mapping); `qwen/variants/edit/qwen_image_edit.py` guidance>1.0 negative-prompt path; `flux/variants/txt2img/__init__.py` re-exports `Flux1`.
8. **Single-file layouts at snapshot root** in `path_resolution.py` (`model.safetensors`/`transformer.safetensors` for `transformer`, `ae.safetensors`/`vae.safetensors` for `vae`).
9. **`models/flux3/` tree** (ours, kept wholesale) + `flux3_norm_out.py` + split BFL vs Diffusers VAE mappings (`get_bfl_vae_mapping()` vs `Flux2WeightMapping.get_vae_mapping()`) and optional `.scales`/`.biases` targets in `flux2_weight_mapping.py` & `qwen21_weight_mapping.py` for mflux 0.24's `missing_required_names()` validation.
10. **Qwen-Image 2.1 VAE decode** (`qwen21_vae/blocks.py` and `vae.py`): cast decoder convolution inputs to their weight dtype, keeping FP32 latent denormalization and the existing tiling policy. Encoding and Qwen-Image Edit are unchanged.

## Re-pinning the guard

`backend/test_repo_security.py::test_vendored_mflux_is_unmodified` pins the file
count and a SHA-256 over path and size. After an intentional vendor edit, compute
the new values with the same loop as the test.
