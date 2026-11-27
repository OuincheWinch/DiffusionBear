<p align="center">
  <a href="https://github.com/OuincheWinch/DiffusionBear/releases/tag/v0.5.2">
    <img src="docs/screenshots/0.5.2/052_annonce.webp" alt="DiffusionBear v0.5.2 — Local AI Image Studio for Apple Silicon" width="100%">
  </a>
</p>

<h1 align="center">DiffusionBear</h1>

<p align="center"><strong>v0.5.2</strong> &nbsp;·&nbsp; a local-first image generation studio for Apple Silicon</p>

<p align="center">
  <a href="https://www.diffusionbear.com"><strong>Website</strong></a> &nbsp;·&nbsp;
  <a href="https://www.diffusionbear.com/agentic-api.html"><strong>Agentic API</strong></a> &nbsp;·&nbsp;
  <a href="https://www.youtube.com/watch?v=D6Nq4XPxHf8&t=1s"><strong>Video tutorial</strong></a> &nbsp;·&nbsp;
  <a href="https://x.com/DiffusionBear"><strong>x.com/DiffusionBear</strong></a> &nbsp;·&nbsp;
  <a href="https://diffusionbear.com/discord"><strong>Discord</strong></a>
</p>

<p align="center">
  <a href="https://github.com/OuincheWinch/DiffusionBear/releases/tag/v0.5.2"><img src="https://img.shields.io/badge/release-v0.5.2-blue" alt="Latest release v0.5.2"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT license"></a>
  <a href="https://github.com/ml-explore/mlx"><img src="https://img.shields.io/badge/engine-MLX%20%2F%20Metal-orange" alt="Built on MLX / Metal"></a>
  <img src="https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey" alt="macOS, Apple Silicon only">
  <img src="https://img.shields.io/badge/privacy-100%25%20local%20%C2%B7%20no%20account-7A5AF8" alt="100% local, no account, no token">
</p>

## Quick start

1. **Download** [DiffusionBear-0.5.2-arm64.zip](https://github.com/OuincheWinch/DiffusionBear/releases/download/v0.5.2/DiffusionBear-0.5.2-arm64.zip) (**218 MB**) — or browse the [v0.5.2 release page](https://github.com/OuincheWinch/DiffusionBear/releases/tag/v0.5.2) for the `.dmg` (**288 MB**) and SHA-256 checksums.
2. **Install:** unzip, drag `DiffusionBear.app` into `/Applications`, then run the one-time Gatekeeper bypass `xattr -cr /Applications/DiffusionBear.app` in Terminal (step-by-step guide below).
3. **Generate:** launch the app, pick a model in the **Models** tab (weights download once, then the app works 100% offline), type a prompt, hit **Generate**. No account, no cloud, no per-image billing.


---

**DiffusionBear** — a free macOS studio that runs SDXL (and its derivatives), FLUX.1, FLUX.2 Klein 4B/9B, FLUX.3 Klein 5.6B, MOSS-Image SFT 5B, Kromoss 5B, Qwen-Image-2.1 & Edit & Turbo, Krea 2 Turbo, Z-Image-Turbo, Ming-Image 0.2 Design, Anima Turbo 2B, DreamLite Mobile 0.39B, Bria FIBO 2.0 Edit Turbo and Bria RMBG-3.0 background removal on your Mac's own Metal GPU. No account, no cloud, no per-image
billing.

It is built on Apple's [MLX](https://github.com/ml-explore/mlx) framework, so it is
**Apple Silicon only**. There is no CUDA or CPU path — MLX is macOS/Metal
exclusive.


Made by **[Ouinche](https://www.ouinche.com)** — follow on X:
**[x.com/DiffusionBear](https://x.com/DiffusionBear)** 
— released under the [MIT licence](LICENSE).

DiffusionBear is an independent project and a DiffusionBee alternative. Not affiliated with DiffusionBee.

---

## Made with DiffusionBear

<table>
  <tr>
    <td align="center" width="33%"><img src="docs/showcase/lightbulb-city-z-image-turbo.webp" alt="A miniature city inside a glowing light bulb, generated locally with Z-Image Turbo" width="260"></td>
    <td align="center" width="33%"><img src="docs/showcase/astronaut-sunflowers-z-image-turbo.webp" alt="An astronaut standing in a sunflower field at golden hour, generated locally with Z-Image Turbo" width="260"></td>
    <td align="center" width="33%"><img src="docs/showcase/glowing-mushroom-forest-z-image-turbo.webp" alt="A misty forest with giant glowing mushrooms, generated locally with Z-Image Turbo" width="260"></td>
  </tr>
  <tr>
    <td align="center"><sub>Z-Image Turbo 6B · 6 steps · seed 1006 · 576×1024 · 29.07 s</sub></td>
    <td align="center"><sub>Z-Image Turbo 6B · 6 steps · seed 1004 · 576×1024 · 28.59 s</sub></td>
    <td align="center"><sub>Z-Image Turbo 6B · 6 steps · seed 2024 · 576×1024 · 28.15 s</sub></td>
  </tr>
</table>

<details>
<summary>Prompts and setup</summary>

Unedited outputs (WebP-encoded for this page) from the studio, generated for the DiffusionBear short videos. Times measured on an Apple M4 Pro virtual machine with 16 GB.

1. `macro photo of a tiny miniature city living inside a glowing light bulb, tilt shift, intricate detail, warm light, dark background`
2. `an astronaut in a white spacesuit standing in a huge sunflower field at golden hour, helmet visor reflecting the sunset, dreamy film photography`
3. `misty enchanted forest with giant glowing mushrooms, fantasy, cinematic`

</details>

---

## What's new in v0.5.2

This release focuses on **Apple Silicon memory efficiency, M1 compute speedups, and 8 GB Mac support**, backed by controlled A/B measurements on an Apple M1 (16 GB):

- **FP16 Compute Precision toggle (`BF16` / `FP16 ⚡`) + per-request API override:** instant switch in **Parameters → Engine** across all MLX engines (FLUX.1/2/3, Z-Image, Qwen-Image 2.1, MOSS/Kromoss, FIBO 2.0/RMBG-3.0, Krea 2). Cuts latency by **-10% to -67.9%** on M1 (`compute_precision` can also be overridden per call in `POST /api/generate`).
- **Seamless Row-Strip VAE Decode (`VAEBlockDecoder.LOW_MEMORY`):** active on FLUX.2, FLUX.3, Z-Image, MOSS/Kromoss, and FIBO 2.0/RMBG-3.0 (plus Qwen 2.1 native strips). At `1024×1024` FP16, cuts VAE peak MLX memory by **-79.0% to -82.3%** (`6,860 MB → 1,442 MB` on FLUX.2-4B; `8,144 MB → 1,443 MB` on FLUX.3-5.6B) with **SSIM ≥ 0.9978** and zero tile seams.
- **Qwen-Image 2.1 StepCache (`0.25`, `steps >= 16`):** automatically skips redundant denoising steps on long non-turbo runs, saving **-31.6% to -34.4%** wall time across 25–40 steps (`398.98 s → 261.87 s` at 25 steps; `548.74 s → 371.39 s` at 40 steps) at **SSIM ≥ 0.981**.
- **SSD Stream (`RAM` / `SSD ⚡`, 8 GB Mac mode):** streams Qwen-Image 2.1 weights from SSD instead of holding them in unified memory, cutting peak footprint by **-64.1%** (`11,639.8 MB → 4,181.5 MB`, `SSIM = 1.00000`), qualifying Qwen-Image 2.1 under the 8 GB ceiling.
- **Opt-in FLUX.2-klein 9B Light Text Encoder (`5.1B Q4-G64`):** [`Ouinche/flux2-klein-9b-qwen3-te-pruned-5.1b-mlx-q4g64`](https://huggingface.co/Ouinche/flux2-klein-9b-qwen3-te-pruned-5.1b-mlx-q4g64) (**2.57 GiB**, `-73%` disk vs 9.50 GiB BF16). Opt-in download in **Parameters → Engine** (never forced on existing installs), cuts text-encoder peak MLX RAM by **-14.4% to -15.4%**, with isolated prompt-cache keys (`:small9b`).
- **Ming-Image 0.2 Design 24.9B *(experimental Q4/Q8, MIT)*:** originally targeting 80 GB datacenter GPUs, now plugged into DiffusionBear via a stripped native Swift/MLX helper (`+29 MB` app / `+0.3 MB` zip, reusing the bundled `mlx.metallib`) running strictly in `BF16` with official Figma-style structured JSON prompt rewriting and optional RGBA alpha transparency (**614.32 s** median across 11 benchmark prompts on base M1 16 GB; **~10m 30s – 11m 14s** at native `1024×1024`).
- **Runtime & metrology:** upgraded MLX, Transformers, and mlx-vlm runtime, rebased vendored `mflux` (`6aebc2d6`), and added unified `memory_peaks` / `compute_precision` / `vae_mode` telemetry to generation sidecars.

## What's new in v0.5.1

This release adds three engines and a set of smaller improvements.

- **Anima Turbo 2B (q4):** anime-style engine with Anima LoRAs, a 4-step fast mode and a 10-step quality mode. 24.34 s (4 steps) and 52.75 s (10 steps) median at 512×768 on M1 16 GB.
- **Bria RMBG-3.0 background removal (q4, `Ouinche/RMBG-3.0-MLX-4bit`):** removes the background of an image from the Studio Canvas and keeps the result as a transparent PNG. Three quality levels: Fast (512 px), HQ (768 px) and Native (1:1). CC BY-NC 4.0.
- **DreamLite Mobile 0.39B (q4):** a small 4-step UNet for quick drafts, 4.62 s median at 512×768 on M1 16 GB. CC BY-NC 4.0.
- **MCP server (`mcp-server/`):** Model Context Protocol wrapper around the local REST API — 11 tools (`generate_image`, `create_image`, `list_models`, `list_loras`, `job_status`, `wait_for_image`, …), bundled in the `.app` and installable standalone (`pip install ./mcp-server`). Agents (Claude Desktop, Cursor, Cline) drive DiffusionBear fully offline.
- **ControlNet on MLX, download-on-first-use only:** FLUX.1-schnell Canny (Apache-2.0 `mflux-community/flux-1-schnell-controlnet-canny-mflux-q4`), Qwen-Image-2.1-Turbo ControlNet-Union and Z-Image-Turbo ControlNet-Union-2.1. Weights are fetched **once**, only when a control image is attached on a supporting model and downloads are allowed — never at startup — and per-stage eviction keeps them inside the 16 GB wired budget.
- **Vendored mflux rebase (0.24 line)** with Qwen-Image-2.1 prompt-embedding cache pinning; the staged pipeline now reports per-step progress, UI phases and mid-generation cancellation.
- **Qwen-Image 2.1 RGBA fix:** alpha-channel (transparent) generation no longer crashes in the VAE decode — verified end-to-end with a real RGBA render.
- **Full benchmark suite refreshed:** 22 model configurations × 11 prompts = 242 timed generations on M1 16 GB.
- Additional fixes merged before the next release: accelerators/adapters never download without consent ([#46](https://github.com/OuincheWinch/MLX-Diffusion-DEV/pull/46)), MOSS-Image SFT ≤4-step uses `lvladikov/MOSS-SFT-Distill-4step-LoRA` ([#42](https://github.com/OuincheWinch/MLX-Diffusion-DEV/pull/42)), local models can be added from any absolute folder ([#47](https://github.com/OuincheWinch/MLX-Diffusion-DEV/pull/47)), faster Qwen-Image VAE decode ([#40](https://github.com/OuincheWinch/MLX-Diffusion-DEV/pull/40)) and Krea 2 denoising in bf16 ([#45](https://github.com/OuincheWinch/MLX-Diffusion-DEV/pull/45)).

## What's new in v0.5.0

- **Full deletion of PyTorch & trimmed MLX bundle (`-51%` `.app`, `-45%` `.zip`):** **1.3 GB → 621 MB**
  `.app` (**396.7 MB → 217.7 MB** arm64 `.zip`). `torch`, `torchvision`,
  `torchaudio`, `diffusers`, and `accelerate` are 100% deleted from the
  packaged `.app` and default runtime. Only the exact runtime modules needed are
  bundled (`mlx`, `safetensors`, NumPy, `transformers`, `tokenizers`, and `--no-deps` `mlx-vlm`), compiled with `0` build warnings.
- **Rebuilt 3-Stage / 4-Stage MLX Generation Pipeline:** replaced the monolithic
  pipeline with a strict sequential memory lifecycle so **only one heavy model
  tree is resident in Unified Memory at a time**:
  - **3-stage (Text-to-Image):** **Encode** (Text Encoder → embeddings → evict) → **Generate** (DiT/UNet → latents → evict) → **Decode** (VAE / fast TAEF mini-VAE → pixels → evict).
  - **4-stage (Reference / img2img / Edit):** adds **Stage 1/4 Reference Encode** at the start — loading the VAE (or Vision-Language encoder), encoding reference image(s) into latent space (`image_latents` + `image_latent_ids`), and evicting the reference encoder before prompt encoding, denoising with injected reference latents, and final VAE decoding.
- **Generation speed — v0.4.x suite (historical, superseded by the current table in [Benchmarks](#benchmarks); Apple M1 16 GB, 512×768 medians):** 19-config / 209-run suite
  finishes in **5h 24m 26s** (**-58.5%** vs prior 13h 01m 38s reference, **-4.7%** step-matched):
  - **MOSS-Image SFT 5B (4s DMD, former darask0 kit):** **83.57 s** (**-96.7%** vs 25-step stock)
  - **Qwen-Image 2.1 (2s + DMD2):** **46.77 s** (**-32.0%**) · **Turbo 8s:** **158.64 s** (**-19.4%**)
  - **FLUX.1-schnell (4s):** **74.22 s** (**-16.5%**) · **RealVisXL V5.0 (8s):** **38.78 s** (**-16.0%**)
  - **Juggernaut XL Lightning (4s):** **8.80 s** (**-9.9%**) · **Bria FIBO 2.0 Edit Turbo (4s):** **93.68 s** (**-8.3%**)
  - **FLUX.2-klein 9B (4s):** **96.54 s** (**-8.0%**) · **FLUX.2-klein 4B (4s):** **44.09 s** (**-7.2%**)
  - **SDXL persistent q4 cache:** warm SDXL + LoRA reloads drop from ~34 s to ~3.45 s via staged TE/UNet eviction and persistent prequantized artifacts.
- **Kromoss 5B (4-bit MLX) + 8-step Accelerator LoRA:** added support for **Kromoss 5B** (`Ouinche/kromoss-mlx-q4`, a 4-bit MLX distillation of MOSS / `lodestones/Kromoss`), running natively on the pure-MLX MOSS engine with its automatic **8-step DMD2 Accelerator LoRA** (**129.43 s median** at 512×768).
- **Canonical Model Families & Qwen 2.5 Prompt Enhancers:** every architecture now belongs to its own canonical family (`sdxl`, `flux1`, `flux2`, `flux3`, `krea2`, `z-image`, `qwen`, `moss`, `fibo2`, `ming`) with a dedicated local **Qwen 2.5** (0.5B / 1.5B MLX) prompt-enhancer profile. Ming uses inclusionAI's structured `canvas_settings` + `layers` JSON rewriting contract.
- **Qwen-Image 2.1 & Turbo Alpha Channel (RGBA Transparency):** added a toggle in **Advanced Settings** to force 4th-channel alpha transparency at generation time, with checkerboard preview across Canvas, Browser, and Image Cards.
- **First-run model SHA-256 verification & partial repair:** hashes installed built-in models once on first launch against SHA-256 manifests embedded in the binary (custom user models untouched), flags mismatches from legacy upgrades, and offers shard-level partial or full redownload — also available anytime on the **Models** tab.
- **Open Agentic REST API & Remote Browser Access (`http://127.0.0.1:8001`):**
  drive any installed model, LoRA, Reference Edit, mask, or Virtual Box
  programmatically from local or remote AI agents (`Claude Code`, `Cursor`, `Antigravity`) or scripts — **no cloud tokens, no subscriptions, zero per-image billing** — or generate remotely using **only one web browser** (`Chrome`, `Firefox`, `Opera`, `Safari` on Mac/iPhone/iPad) with managed `dbk_...` keys (`SHA-256`), `Authorization: Bearer` / `X-API-Key`, wildcard bind fail-safe, and `429` queue rate limits. Interactive guide at
  **[diffusionbear.com/agentic-api.html](https://www.diffusionbear.com/agentic-api.html)**
  and step-by-step setup in **[docs/parameters-api-guide.md](docs/parameters-api-guide.md)**.

## What's new in v0.4.5 (beta)

- **Native MLX SDXL engine — rewritten from scratch:** the `mlx_diffuser` dependency is gone. We now own the MLX UNet, dual CLIP encoders, VAE decoder, schedulers, LoRA stack, 4/8-bit quantization, resident text encoders and TAESD. Removing the second `venv-sdxl` runtime cut the app bundle from **1.7 GB to 1.3 GB** and the release zip from **510 MB to 385 MB**.
- **Long prompts and mandatory adapters:** SDXL prompts are split across 77-token CLIP windows instead of silently truncated; Qwen DMD2 and MOSS DMD adapters auto-restore when missing; Qwen 2.1 reference images now warn to use Qwen-Image Edit for instruction edits.
- **Qwen base self-healing:** the base pack now supplies its own tokenizer/scheduler/model_index instead of borrowing another Qwen install; incomplete component weights fail with a clear re-download error instead of rendering a texture.
- **Licence display and validation reviewed.**
- **LoRA management polished, including the Krea 2 Turbo LoRA recipe.**
- **MOSS-Image batching fixed:** the second image in a cached batch no longer fails during VAE decode.
- **MOSS LoRAs fixed:** detected/registered as MOSS, selectable in the architecture dropdown, and the real tensor layout now overrides Civitai metadata.
- **Z-Image / FLUX samplers:** Euler now maps to the flow-match scheduler instead of the Krea 2 scheduler.
- **French model sizes:** shown as `GB` instead of `Go`.
- **Qwen-Image integration refined.**
- **Minor bug fixes.**

## What's new in v0.4.4 (beta)

- **Qwen-Image-2.1 Edit support**, with Qwen-VL reference conditioning and the Edit-specific prompt path.
- **Qwen-Image-2.1 and Krea 2 Turbo:** Hugging Face depot repackaging.
- **Linoy Tsaban 2-step DMD2 Distill LoRA for Qwen-Image 2.1:** high-speed 2-step generation (~33–50s after initial compile) with automatic 2-step/CFG-1 tuning.
- **Gallery Reference Image Filtering:** isolate generations made with visual reference conditioning.
- **Website and documentation upgrades**, typography fixes, unified navigation, and minor bug fixes.

## What's new in v0.4.3 (beta)

- **Browser page revamp:** navigate pages with `←` / `→` and jump directly to a page number.
- **Memory management tweaks:** large generations are checked against the machine's safe megapixel ceiling before they can trigger a Metal out-of-memory crash.
- **Minor bug fixes.**

## What's new in v0.4.2 (beta)

- **Full support of FIBO 2.0 Edit Turbo MLX 4-bit:** First-ever 4-bit quantized port of Bria AI's state-of-the-art instruction editing model running natively on Apple Silicon via MLX ([Hugging Face: Ouinche/FIBO2-Edit-Turbo-MLX-4bit](https://huggingface.co/Ouinche/FIBO2-Edit-Turbo-MLX-4bit) | [Original Model: briaai/Fibo-2.0-edit-turbo](https://huggingface.co/briaai/Fibo-2.0-edit-turbo)).
- **Text-Guided Reference Edit:** High-fidelity single reference editing with natural language prompts in only 4 denoising steps (~85s on M1).
- **Masked Edit (Precision Inpainting):** Interactive brush painting directly on reference image backdrop in DiffusionBear Studio. Zero variance outside masked area with automated 1.5px Gaussian edge blending and unmasked composite retention.
- **Multiple References (Multimodal In-Context Editing):** Combine up to 4 reference images (e.g., character portrait + specific dress or object) grounded through the integrated 4-bit Qwen3-VL vision-language encoder.
- **3-Stage Sequential Memory Management:** Lazy component lifecycle (Qwen3-VL → DiT Transformer → FLUX.2 VAE) keeping resident VRAM strictly below 6.8 GB on 16 GB Apple Silicon.
- **Visual Showcase & Benchmarks:** Interactive comparison suite, side-by-side comparisons with original Bria outputs, and telemetry report at [diffusionbear.com/fibo.html](https://www.diffusionbear.com/fibo.html).

## What's new in v0.4.1 (beta)

- **SDXL Metal Memory Stabilization:** Systematic Metal buffer clearing and working-set memory management (+51.9% suite-wide speedup, 4.25x faster on Juggernaut XL Lightning down to 9.2s).
- **Generation Pipeline Optimization:** Lossless JIT compilation (`@mx.compile`), sub-second TAESD VAE decoding, and `euler_trailing` timesteps restoring 100% visual parity (&infin; dB PSNR).
- **Interactive 187-Run Benchmark Arena:** Explore all 17 model configurations with our interactive comparison slider at [diffusionbear.com/benchmark.html](https://www.diffusionbear.com/benchmark.html).

## What's new in v0.4.0 (beta)
- **Virtual Boxes on the Studio Canvas:** Draw, resize, and label color-coded spatial bounding boxes right on the canvas for **FLUX.3-klein 5.6B** and **MOSS-Image SFT 5B** to position subjects, characters, and exact poster typography without ControlNet.
- **Smart 16 GB Unified Memory Management:** 
- **Gallery Favorites & Privacy Card Blur:** 
- **Custom Python Samplers:** 
- **DiffusionBee & Draw Things Local Library Import:** if you allow DiffusionBear to read theirs directories
---

## Supported models

**Fifteen+ checkpoints across twelve families**, 4-bit quantised and memory-tuned for 16 GB of unified memory. Switching engines swaps weights on demand rather than keeping everything resident.

| Model | Family | Default steps | What it supports |
|---|---|---|---|
| **Anima Turbo 2B** | Anima | 4–10 | Lightweight 2B anime engine (q4, 3.1 GB) with Anima LoRAs and `er_sde`/`euler` samplers; 4-step fast mode and 10-step quality mode. Weights are non-commercial; generated images may be used commercially. |
| **FIBO 2.0 Edit Turbo** | FIBO 2.0 | 4 | Up to 4 reference images, 4-step distilled editing, guidance-free (no negative prompt), and a Qwen3-VL 4B text encoder. |
| **FLUX.1-schnell** | FLUX.1 | 4 | Classic 4-step rectified flow, 1 reference image, FLUX.1 LoRAs, fast-VAE decode, and **ControlNet-Canny** (control image + strength in Generate → Advanced; branch downloaded once on first use). |
| **FLUX.2-klein 4B** | FLUX.2 | 4 | Up to 10 reference images, FLUX.2 LoRAs, and generative fill. Guidance-distilled (no negative prompt). |
| **FLUX.2-klein 9B** | FLUX.2 | 4 | 9-billion-parameter flagship DiT with up to 10 reference images, LoRAs, and superior detail. |
| **FLUX.3-klein 5.6B** | FLUX.3 | 16 | 16-step flow-matching distilled, up to 10 reference images, and interactive **Virtual Boxes** on the Studio Canvas. |
| **DreamLite Mobile 0.39B** | DreamLite | 4 | The « Instantané » mode: a 0.39B UNet distilled to 4 guidance-free steps (4.62 s median at 512×768 on M1 16 GB), no LoRAs/references. CC BY-NC 4.0. |
| **Krea 2 Turbo 13B** | Krea 2 | 4–8 | 13B photorealistic engine with 1 reference image, LoRAs, persistent Q4 text-encoder cache, and auto-applied 4-step distilled LoRA. |
| **Kromoss 5B** | MOSS | 8 | 4-bit MLX distillation of MOSS with automatic **8-step DMD2 Accelerator LoRA** (`8-steps-dmd2-lora.safetensors`), Virtual Boxes, and pure-MLX staged execution. |
| **Ming-Image 0.2 Design 24.9B** *(experimental)* | Ming-Image | 12 | Typography/UI/design generation with exact-text prompts (24.9B total: 6.15B DiT plus a 17.8B text-conditioning stack) and official Figma-style structured prompt rewriting, optional **RGBA alpha-channel transparency**, official 1024² fast-square plus 2K 1:1/16:9/4:3/2:3 presets, and staged q4/q8 MLX execution through a pinned Swift helper. No LoRAs or reference images in the first integration. |
| **MOSS-Image SFT 5B** *(+ Base 5B)* | MOSS | 4–8 (DMD) / 25 | Interactive **Virtual Boxes**, reference image, LoRAs, MLX q8-g64 text encoder, and **4-step (`⚡ DMD 4`) / 8-step (`⚡ DMD 8`)** step-distillation adapters (fetched only with your consent). |
| **Bria RMBG-3.0** | FIBO 2.0 (utility) | 4 | Background removal from the Studio Canvas to a transparent PNG, with Fast (512 px), HQ (768 px) and Native (1:1) quality levels. CC BY-NC 4.0. |
| **Qwen-Image Edit** | Qwen-Image | 8–25 | Instruction-based editing with up to 3 reference images, negative prompts and guidance, plus 8/15/20/25-step presets. No 2-step adapter (the 60-layer edit transformer is incompatible with Qwen-Image 2.1 DMD2). |
| **Qwen-Image-2.1** | Qwen-Image | 2–25 | 7B single-stream DiT with Qwen3-VL text encoder, 1 reference image, negative prompts, optional **RGBA alpha-channel transparency** in Advanced Settings, and an auto-injected **2-step DMD2 LoRA** (`linoyts/Qwen-Image-2.1-DMD2-2steps`, ~33–50 s). |
| **Qwen-Image-Turbo-2.1** | Qwen-Image | 4–8 | 8-step distilled Qwen-Image 2.1 variant for 16 GB Macs, with optional **RGBA alpha-channel transparency**, its own 4-step distillation LoRA, and **ControlNet-Union** support (control image + strength in Generate → Advanced, LoRAs supported). |
| **SDXL & Derivatives** *(Juggernaut XL Lightning, RealVisXL V5.0 Lightning / Hyper-SD, Juggernaut XI v11, Custom SDXL / Illustrious)* | SDXL | 4–50 | 4-step Lightning (8.19 s median on M1 16 GB) or 20–50 step custom `.safetensors` checkpoints (auto-converted), negative prompts, sampler picker, CLIP Skip, DeepCache, and TAESD fast decode. |
| **Z-Image-Turbo 6B** | Z-Image | 6–8 | Compact 6B engine with 1 reference image, LoRAs, fast VAE, crisp architectural geometry, and **ControlNet-Union 2.1** (depth/canny/pose hints, control image + strength in Generate → Advanced; branch downloaded once on first use). |

> **Recommended fast path:** Qwen-Image 2.1 at 2 steps with the `linoyts/Qwen-Image-2.1-DMD2-2steps` LoRA (downloaded once, after you allow it, guidance locked to 1.0). Qwen-Image Turbo 2.1 uses its separate 4-step LoRA and is slightly slower. Qwen-Image Edit has no 2-step adapter.

**Bring any SDXL or Illustrious checkpoint you own:** single-file Civitai `.safetensors` checkpoints are converted in place, unlisted models can be onboarded from a local folder or your Hugging Face cache, and the app also browses models and LoRAs already downloaded by **Draw Things** and **DiffusionBee** (strictly read-only).

---

## Studio Interface Screenshots

| Studio Generation (Dark Mode) | Studio Generation (Light Mode) |
|---|---|
| ![DiffusionBear Studio generation interface in dark mode](docs/screenshots/0.5.2/052_en_generate_dark.webp) | ![DiffusionBear Studio generation interface in light mode](docs/screenshots/0.5.2/052_en_generate_light.webp) |

| Gallery / Browser (Dark Mode) | Gallery / Browser (Light Mode) |
|---|---|
| ![DiffusionBear Gallery with search and favorites in dark mode](docs/screenshots/0.5.2/052_en_browser_dark.webp) | ![DiffusionBear Gallery with search and favorites in light mode](docs/screenshots/0.5.2/052_en_browser_light.webp) |

| Models: Installed & Integrity Check | Models: Hugging Face Browser |
|---|---|
| ![DiffusionBear Models tab and installed model management](docs/screenshots/0.5.2/052_en_models_installed_dark.webp) | ![DiffusionBear Hugging Face model browser](docs/screenshots/0.5.2/052_en_models_browse_light.webp) |

| Parameters: Engine, GPU Memory & Model Defaults | Parameters: Language & System Preferences |
|---|---|
| ![DiffusionBear Parameters tab with GPU memory and custom model defaults](docs/screenshots/0.5.2/052_en_parameters_preferences_dark.webp) | ![DiffusionBear Parameters preferences in light mode](docs/screenshots/0.5.2/052_en_parameters_preferences_light.webp) |

| Parameters: API Keys & Remote Access | Parameters: Qwen 2.5 Prompt Enhancer |
|---|---|
| ![DiffusionBear Parameters API keys and remote connection](docs/screenshots/0.5.2/052_en_parameters_api_dark.webp) | ![DiffusionBear Parameters Qwen 2.5 prompt enhancer](docs/screenshots/0.5.2/052_en_parameters_enhancer_dark.webp) |

| Models: Detected on Disk | Models: Import from Draw Things / DiffusionBee |
|---|---|
| ![DiffusionBear models detected on disk](docs/screenshots/0.5.2/052_en_models_detected_dark.webp) | ![DiffusionBear import from Draw Things and DiffusionBee](docs/screenshots/0.5.2/052_en_models_import_dark.webp) |

---

## Benchmarks

Measured on a **2021 MacBook Pro M1 (16 GB)** running **FP16 mode** across **23 model configurations × 11 prompts = 253 timed studio generations** (22 core FP16 configs / 242 runs in **4h 23m 49s**, plus experimental Q4 **Ming-Image 0.2 Design 24.9B** in BF16 via Swift/MLX helper / 11 runs in **1h 55m 04s**, total **6h 18m 53s**; 10 portrait prompts at `512×768` + 1 icon prompt at `512×512`; across the **18 matched configs / 198 runs**: **-69.3%** total / **-27.4%** step-matched vs reference):

| Model | Steps | M1 16 GB FP16 median (`512×768`) | M1 16 GB range | M4 Pro VM 16 GB median (`512×768`) | Family / Notes |
|---|---|---|---|---|---|
| **DreamLite Mobile 0.39B** | 4 | **3.67 s** | 3.61 – 5.69 s | 2.24 s | DreamLite · 0.39B mobile distilled, q4 (`-20.6%` vs BF16) |
| **Juggernaut XL Lightning** | 4 | **9.73 s** | 9.28 – 20.55 s | not measured | SDXL · Native MLX, q4 cache + TAESD |
| **RealVisXL V5.0 Lightning** | 6 | **12.80 s** | 12.64 – 21.84 s | not measured | SDXL · 6-step Lightning, native MLX + TAESD |
| **Anima Turbo 2B** | 4 | **14.21 s** | 14.17 – 14.87 s | 7.81 s | Anima · 2B anime turbo, q4, er_sde (`-41.6%` vs BF16) |
| **Anima Turbo 2B** | 10 | **34.31 s** | 34.10 – 37.44 s | 13.39 s | Anima · 2B anime turbo 10-step, q4, er_sde (`-35.0%` vs BF16) |
| **FLUX.2-klein 4B** | 4 | **34.34 s** | 34.23 – 38.70 s | 10.11 s | FLUX.2 · Compact 4B rectified flow (`-21.8%` vs BF16) |
| **FLUX.3-klein 5.6B** | 4 | **37.69 s** | 37.49 – 42.20 s | 14.04 s | FLUX.3 · 5.6B flow-matching + Virtual Boxes (`-21.0%` vs BF16) |
| **Juggernaut XI v11** | 8 | **38.26 s** | 38.11 – 51.62 s | not measured | SDXL · Hyper-SD 8-step, native MLX |
| **RealVisXL V5.0** | 8 | **39.23 s** | 37.96 – 49.26 s | not measured | SDXL · Hyper-SD 8-step, native MLX |
| **Qwen-Image 2.1 (2s + DMD2)** | 2 | **59.39 s** | 37.26 – 77.38 s | 17.48 s | Qwen-Image · 7B DiT + 2-step DMD2 |
| **FLUX.2-klein 4B (8s)** | 8 | **63.50 s** | 63.40 – 63.99 s | 19.05 s | FLUX.2 · High-detail 8-step mode (`-24.3%` vs BF16) |
| **MOSS-Image SFT 5B (DMD 4s)** | 4 | **67.58 s** | 66.71 – 72.71 s | not measured | MOSS · Pure MLX 5B DiT + 4-step lvladikov LoRA (`-13.2%` vs BF16) |
| **FLUX.2-klein 9B** | 4 | **69.47 s** | 68.92 – 77.56 s | 20.16 s | FLUX.2 · Flagship 9B rectified flow (`-23.5%` vs BF16) |
| **FLUX.3-klein 5.6B (8s)** | 8 | **71.26 s** | 71.05 – 71.76 s | 25.55 s | FLUX.3 · High-detail 8-step mode (`-22.8%` vs BF16) |
| **FLUX.1-schnell** | 4 | **78.25 s** | 77.90 – 82.36 s | 20.93 s | FLUX.1 · Classic 4-step rectified flow |
| **Bria FIBO 2.0 Edit Turbo** | 4 | **79.81 s** | 79.54 – 82.85 s | 23.08 s | FIBO 2.0 · 4.2B DiT + Qwen3-VL 4B (`-8.0%` vs BF16) |
| **Z-Image Turbo 6B** | 8 | **91.12 s** | 90.43 – 102.51 s | 26.59 s | Z-Image · Sharp geometry 8-step (`-23.1%` vs BF16) |
| **Qwen-Image 2.1 Turbo** | 8 | **106.17 s** | 105.46 – 123.53 s | 41.33 s | Qwen-Image · 8-step distilled 7B DiT (`-27.2%` vs BF16) |
| **Kromoss 5B (DMD2 8s)** | 8 | **111.70 s** | 110.70 – 121.27 s | 37.90 s | MOSS · Pure MLX 5B DiT + 8-step DMD2 LoRA (`-12.0%` vs BF16) |
| **Krea 2 Turbo 13B** | 4 | **117.86 s** | 113.62 – 137.58 s | 31.85 s | Krea 2 · 13B photorealistic 4-step distilled (`-25.8%` vs BF16) |
| **FLUX.2-klein 9B (8s)** | 8 | **137.14 s** | 136.67 – 137.63 s | 40.02 s | FLUX.2 · Flagship 9B 8-step (`-23.8%` vs BF16) |
| **Krea 2 Turbo 13B (8s)** | 8 | **186.22 s** | 184.40 – 196.17 s | 63.86 s | Krea 2 · 13B native 8-step (`-23.5%` vs BF16) |
| **Ming-Image 0.2 Design 24.9B** *(exp. Q4)* | 12 | **614.32 s** | 591.76 – 744.90 s | not measured | Ming-Image · Experimental Q4/Q8 (`BF16`), plugged via Swift/MLX helper (`1024px` internal bucket) |

The **M4 Pro VM** column is secondary: an Apple M4 Pro **virtual machine** with 16 GB, measured on the prior release, **17 configurations × 11 prompts = 187 timed generations**, same prompts and sizes. The four SDXL configurations and MOSS-Image SFT were not measured there. A virtual machine is not a retail Mac, so treat this column as indicative.

The MOSS-Image SFT 4-step row was measured with the former darask0 4-step kit (removed from Hugging Face); the current ≤4-step path uses the lvladikov LoRA and has not been re-benchmarked yet.

Sources: [`docs/Comparaison040/benchmark_data.json`](docs/Comparaison040/benchmark_data.json) (M1, per-run times) and [`docs/Comparaison040/benchmark_data.m4pro_vm.json`](docs/Comparaison040/benchmark_data.m4pro_vm.json) (M4 Pro VM).

For the full interactive visual duel arena, per-prompt latency deltas (`-green` / `+red` vs reference), and historical September baselines, visit **[diffusionbear.com/benchmark.html](https://www.diffusionbear.com/benchmark.html)**.

### FP16 & Memory Optimization Highlights (Apple M1 16 GB)

Measured on the same 2021 MacBook Pro M1 (16 GB) at `512×768` (or `1024×1024` for Row-Strip VAE), identical seeds. Full interactive before/after sliders are on **[diffusionbear.com/fp16-acceleration.html](https://www.diffusionbear.com/fp16-acceleration.html)**.

| Optimization / Model | Baseline | Optimized | Measured Gain | Parity (SSIM / PSNR) |
|---|---:|---:|---:|---:|
| **FP16 · Qwen-Image 2.1 Turbo (2s)** | 108.59 s (BF16) | **34.88 s** (FP16) | **-67.9%** | 0.8942 / 23.73 dB |
| **FP16 · MOSS-Image SFT 5B (4s)** | 107.40 s (BF16) | **65.50 s** (FP16) | **-39.0%** | 0.9885 / 35.95 dB |
| **FP16 · FLUX.2-klein 4B (4s txt2img)** | 83.27 s (BF16) | **58.31 s** (FP16) | **-30.0%** (-1,669 MB RSS) | 0.9613 / 27.99 dB |
| **FP16 · FLUX.2-klein 9B (4s ref)** | 351.90 s (BF16) | **255.98 s** (FP16) | **-27.3%** (-1,221 MB RSS) | 0.9604 / 31.29 dB |
| **FP16 · FLUX.3-klein 5.6B (4s ref)** | 175.34 s (BF16) | **131.65 s** (FP16) | **-24.9%** | 0.9802 / 34.66 dB |
| **FP16 · Bria FIBO 2.0 Turbo (4s ref)** | 119.37 s (BF16) | **90.06 s** (FP16) | **-24.6%** | 0.9991 / 48.32 dB |
| **Row-Strip VAE · FLUX.3 5.6B (`1024×1024`)** | 8,144 MB peak | **1,443 MB peak** | **-82.3% VAE peak** | 0.9978 / 45.84 dB |
| **Row-Strip VAE · FLUX.2 4B (`1024×1024`)** | 6,860 MB peak | **1,442 MB peak** | **-79.0% VAE peak** | 0.9981 / 49.76 dB |
| **StepCache · Qwen 2.1 (25s / 40s)** | 398.98 s / 548.74 s | **261.87 s / 371.39 s** | **-34.4% / -32.3%** | 0.9835 / 0.9885 |
| **SSD Stream · Qwen 2.1 Turbo (8 GB)** | 11,639.8 MB peak | **4,181.5 MB peak** | **-64.1% footprint** | 1.0000 / 99.00 dB |

## Install the App & First-Launch Gatekeeper Guide

`DiffusionBear.app` is a self-contained macOS application: it bundles the backend, one relocatable Python/MLX runtime, and the studio interface inside one `.app`. **You do not need to install Python, Node, Xcode, or Homebrew.**

**[⬇ Download DiffusionBear for macOS (Apple Silicon)](https://github.com/OuincheWinch/DiffusionBear/releases/download/v0.5.2/DiffusionBear-0.5.2-arm64.zip)** &nbsp;·&nbsp; [v0.5.2 Release Notes & Checksums](https://github.com/OuincheWinch/DiffusionBear/releases/tag/v0.5.2)

| | Detail |
|---|---|
| **Requires** | An Apple Silicon Mac (M1 / M2 / M3 / M4 / M5 / M6), **macOS 26.2 or later**, 16 GB unified memory recommended. |
| **Your files** | Settings, gallery, LoRA registry, and model store live in `~/Library/Application Support/DiffusionBear`. To put the store on an external drive, write an absolute path into `~/Library/Application Support/DiffusionBear/store_path`. |
| **Model weights** | Downloaded on first use from Hugging Face or [Civitai](https://civitai.com/?ref_code=88C8VEBA) into your local store. After that, the app works 100% offline. |
| **Package size** | **217.7 MB** arm64 `.zip` → **621 MB** standalone `.app` (pure MLX, no PyTorch or `diffusers`). |

**SHA-256 (zip):** `fda8b2e03f28a6c641e36173efd78c1486d211d06cfce979a79a44aad4a2747c` (all checksums recorded in [packaging/ALL_SHA256SUMS](packaging/ALL_SHA256SUMS)).

### 🍎 How to Open on First Launch (macOS Gatekeeper Bypass)

Because DiffusionBear is a free, open-source project signed ad-hoc (without a paid Apple Developer ID notarization certificate), macOS Gatekeeper pauses the very first double-click of a downloaded copy with this dialog:

<p align="center">
  <img src="docs/screenshots/gatekeeper-blocked.png" alt="macOS Gatekeeper alert — DiffusionBear Not Opened" width="340">
</p>

> **"DiffusionBear" Not Opened** — *Apple could not verify "DiffusionBear" is free of malware that may harm your Mac or compromise your privacy.*

**Follow this one-time 60-second procedure:**

1. **Unzip and move to `/Applications`:** Unzip `DiffusionBear-0.5.2-arm64.zip` and drag **`DiffusionBear.app`** into your **/Applications** folder *(do not launch it from `~/Downloads` or inside the zip)*.
2. **Bypass Gatekeeper (choose Option A or Option B):**
   * **Option A — Fastest (Terminal 1-Liner):** Open **Terminal.app**, paste this command, and press Return:
     ```bash
     xattr -cr /Applications/DiffusionBear.app
     ```
     Then double-click **`DiffusionBear.app`** in `/Applications` — it opens right away!
   * **Option B — No Terminal (System Settings GUI):**
     1. Click **Done** on the Gatekeeper dialog.
     2. Open **System Settings → Privacy & Security**.
     3. Scroll down to the **Security** section (*"DiffusionBear was blocked to protect your Mac"*).
     4. Click **Open Anyway**, authenticate with Touch ID or your password, and click **Open**.

Once authorized, macOS remembers your choice and **never shows the alert again**.

---

## Virtual boxes — place things where you want them

Three engines (FLUX.3-klein, MOSS-Image SFT and Base) accept **spatial boxes**:
instead of hoping the model puts the cat in the right corner, you draw a box on
the Studio Canvas and say what belongs inside it. The canvas can show a
reference image as a backdrop, so you compose over the actual picture you want.

- **SPATIAL BOX panel:** presets (Center / Left / Right / Top / Bottom / full
  frame), free boxes you drag on the canvas, and a live count.
- **Compose** turns your caption + boxes into the layout description the engine
  understands; **JSON** shows exactly what will be sent.
- **Reference backdrops:** overlay any gallery image or reference behind the
  boxes, hide it or remove it at will.
- The layout travels with the generation metadata, so reusing parameters
  reproduces the composition.

For FLUX.2-klein, the equivalent is **generative fill**: paint a mask on a
gallery image and regenerate just that region.

---

## Memory management, quietly handled

Running 4–13 B parameter models on a 16 GB Mac is a memory-budget problem. The
app manages it; you watch (Parameters → Engine & GPU) rather than tune:

- **Auto wired-limit.** Each engine's GPU wired budget is derived from your
  machine — `min(68% of unified memory, Apple's recommended working set)`, about
  **10.9 GB on a 16 GB Mac**. An explicit number you set still wins; `0` means
  unbounded.
- **Idle auto-release.** Resident pipelines (a full FLUX/Krea2 pipeline is
  ~10 GB) are released after **5 minutes of inactivity** (300 s default,
  tunable per engine; `0` = never release). Reloading afterwards costs a few
  seconds, not minutes — macOS keeps the weight files in its file cache.
- **Quantised everything.** Weights are 4-bit by default; the MOSS text encoder
  is swapped for the app's own **MLX q8g64 build (~2.2 GB)** and the Krea 2
  encoder is quantised once and persisted, so later sessions skip both the
  7.5 GB bf16 load and the quantisation pass.
- **Live panel.** Metal memory in use, peak, the wired limits, which pipelines
  are resident, and the idle countdowns — all visible, nothing hidden.
- **Storage cleanup.** The Parameters tab reports generated-image size, TAESD
  state and settings, and the Models tab clears per-repo Hugging Face cache
  entries to reclaim disk space.

### Engine & Performance Toggles (Parameters → Engine)

Three opt-in / instant switches in **Parameters → Engine** (plus two automatic pipeline optimisations) let you tune speed vs memory for your Mac:

| Control | Default | Effect (measured on M1 16 GB) | Recommended for & Limitations |
|---|---|---|---|
| **Compute Precision (`BF16` / `FP16 ⚡`)** | `BF16` *(or `FP16` via settings/API)* | Runs MLX compute in `float16` instead of `bfloat16`: **-10% to -67.9%** faster on M1, up to **-1,669 MB** RSS. Overridable per call via `compute_precision` in `POST /api/generate`. | **M1 / M2 Macs** benefit most (M3/M4 have native hardware BF16). **Limitation:** large 13B models (`Krea 2 Turbo`) can show noticeable composition drift in FP16 (`SSIM 0.7247`); switch back to `BF16` for strict parity. |
| **SSD Stream (`RAM` / `SSD ⚡`)** | `RAM` (Off) | Streams Qwen-Image 2.1 weights on demand from SSD instead of pinning them in RAM: **-64.1% peak footprint** (`11,639.8 MB → 4,181.5 MB`), `SSIM = 1.00000`. | **8 GB Apple Silicon Macs.** **Limitation:** applies to Qwen-Image 2.1 non-edit txt2img runs only (disabled on `qwen-image-edit` reference runs) and requires fast internal NVMe SSD. |
| **FLUX.2-9B Light Text Encoder (`5.1B Q4-G64`)** | Off (opt-in) | Swaps the 9.50 GiB BF16 Qwen3-8B encoder for [`Ouinche/flux2-klein-9b-qwen3-te-pruned-5.1b-mlx-q4g64`](https://huggingface.co/Ouinche/flux2-klein-9b-qwen3-te-pruned-5.1b-mlx-q4g64) (**2.57 GiB**, `-73%` disk, **-14.4% to -15.4%** TE peak MLX RAM). Never auto-downloaded. | **8–16 GB Macs** short on RAM or disk. **Limitation:** Q4 dequantization adds **~3.4–3.8 s** on cold prompt-cache misses and slightly shifts secondary object placement (`SSIM 0.47–0.55` vs 8B, equal visual quality). |
| **Row-Strip VAE Decode** *(automatic)* | On | Decodes latents in row strips with full-image GroupNorm stats (`VAEBlockDecoder.LOW_MEMORY`): **-79% to -82.3%** VAE MLX peak at `1024×1024`, **SSIM ≥ 0.9978**. | Automatic on FLUX.2, FLUX.3, Z-Image, MOSS/Kromoss, FIBO 2.0/RMBG-3.0, and Qwen 2.1. **Limitation:** ~2–3% decode overhead at small `512×768` frames, huge win at `≥1024px`. |
| **Qwen StepCache** *(automatic)* | On (`steps >= 16`) | Caches and skips redundant transformer steps (`step_cache_ratio=0.25`) on non-turbo Qwen 2.1 runs: **-31.6% to -34.4%** faster at 25–40 steps, **SSIM ≥ 0.981**. | Automatic for Qwen-Image 2.1 when `steps >= 16` (disabled below 16 steps where 0 steps are skipped, and disabled on Qwen-Edit / ControlNet). |

### MOSS-Image step distillation (default behaviour)

The **DMD distillation (MOSS)** control in Parameters has four positions:
**Auto**, **DMD 4**, **DMD 8** and **Off**.

| Mode | Adapter used | Schedule |
|---|---|---|
| **Auto** *(default)* | follows the step slider: ≤4 steps → 4-step LoRA, ≤8 → 8-step DMD kit, above → none | as the forced mode below; full CFG trajectory above 8 steps |
| **DMD 4** | lvladikov's 4-step LoRA ([`lvladikov/MOSS-SFT-Distill-4step-LoRA`](https://huggingface.co/lvladikov/MOSS-SFT-Distill-4step-LoRA), `moss_sft_4step_rank_64_lora.safetensors`, rank 64, Apache-2.0, [#42](https://github.com/OuincheWinch/MLX-Diffusion-DEV/pull/42)) | plain LoRA on the stock sampler, run set to 4 steps at CFG 1 |
| **DMD 8** | lodestones' 8-step DMD kit ([`lodestones/MOSS-Image-SFT-8-steps-dmd`](https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd), Apache-2.0) | CFG-free, engine clamps the run to the 8 dataspace codes |
| **Off** | none | full CFG trajectory at the requested steps (~20–45 min) |

The 8-step kit is three files that only work together: a projection **LoRA**, a
distilled fp32 **time embedding**, and a **dataspace** file holding the CFG-free
step schedule. The 4-step LoRA is a single file with no siblings.

With **DMD 4** (or Auto at ≤4 steps), the app uses the 4-step LoRA when it is
installed. If it is missing and no download is allowed, it uses the 8-step kit
when that one is installed instead.

Adapters are never downloaded silently ([#46](https://github.com/OuincheWinch/MLX-Diffusion-DEV/pull/46)).
When the needed adapter is missing, the run stops and explains why. You then
have three options:
- tick **Allow required downloads** in Parameters;
- click **Allow this download and retry**, which allows the download for that
  generation only;
- set the control to **Off**.

The former darask0 4-step kit, now removed from Hugging Face, measured **83.57 s
median** on a 16 GB M1 at 512×768 (**60.45–84.95 s** across 512×512–512×768),
against ~41 min stock. The current 4-step LoRA path has not been re-benchmarked yet.

---

## Agentic Local REST API — Build & Script Locally

DiffusionBear exposes a complete, unthrottled local REST engine at
`http://127.0.0.1:8001` so coding agents (`Claude Code`, `Cursor`, `Antigravity`),
design scripts, and asset pipelines can generate icons, illustrations, edited
mockups, and spatially arranged layouts straight on your Mac's GPU without cloud
keys or per-image billing.

**Full interactive guide & copy-paste agent system prompt:**
**[diffusionbear.com/agentic-api.html](https://www.diffusionbear.com/agentic-api.html)**
&nbsp;·&nbsp; Raw LLM spec:
[site/assets/agentic-api-reference.md](site/assets/agentic-api-reference.md)

**MCP server:** [`mcp-server/`](mcp-server/README.md) wraps this REST API as a
proper **MCP (Model Context Protocol) server** — tools `list_models`,
`generate_image`, `wait_for_image`, `save_image`, … over stdio — so Claude
Desktop / Cursor / Cline can generate locally. Install:
`pip install ./mcp-server` (or `uvx --from git+https://github.com/OuincheWinch/DiffusionBear.git#subdirectory=mcp-server diffusionbear-mcp`),
then set `DIFFUSIONBEAR_API_URL` / `DIFFUSIONBEAR_API_KEY` in the client's
`mcpServers` config. See [mcp-server/README.md](mcp-server/README.md).

**Three-call workflow:**

1. `GET /api/models` — inspect installed models, default dimensions/steps, and capabilities
2. `POST /api/generate` — enqueue a job (`prompt`, `model`, `steps`, `width`, `height`, `loras`, `reference_images`, `mask`, `spatial_boxes`) → returns `{"job_id": "..."}` (HTTP `429` if queue cap is reached)
3. `GET /api/jobs/{job_id}` → poll until `"status": "done"`, then fetch raw bytes from `GET /api/images/{image_id}/file` (or JSON metadata from `GET /api/images/{image_id}`)

```bash
curl -s -X POST http://127.0.0.1:8001/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"model":"flux2-klein-4b","prompt":"flat geometric espresso bear emblem, clean cream paper","steps":4,"width":512,"height":512}'
```

- **Reference edits, inpainting & LoRAs:** upload reference PNGs via
  `POST /api/uploads`, pass `reference_images` + `reference_strength` (or base64
  PNG `mask`), and pre-check LoRA architecture fit with
  `GET /api/loras/{name}/compatibility` and `POST /api/generate/steps/preflight`.
- **Hardened local security (previous release):** `LocalHostMiddleware` and
  `LocalAPIMiddleware` restrict traffic to loopback clients by default; enabling
  API Mode via `/api/security/mode` + `/api/security/keys` enforces strong token
  authentication (`Authorization: Bearer`, `X-MLX-API-Token`, or `X-API-Token`)
  before allowing requests.

---

## Local by design

- **Generation runs on the Metal GPU** through MLX (`mflux` for the
  flow-matching families, a native MLX daemon for SDXL, dedicated engines for
  MOSS and Qwen). Weights are 4-bit so a 4–13 B model fits a 16 GB machine.
- **100% PyTorch-free runtime:** `torch`, `torchvision`, `torchaudio`,
  `diffusers`, and `accelerate` are completely removed from the app bundle and
  default backend — every engine executes via MLX, `safetensors`, NumPy,
  `transformers`, and `tokenizers`.
- **The prompt enhancer is a local LLM too** (Qwen2.5 0.5B or 1.5B, MLX) —
  rewriting a prompt never leaves the machine.
- **No telemetry, no analytics, no network calls at generation time.** Once the
  weights are cached the app works fully offline.
- Tokens ([Civitai](https://civitai.com/?ref_code=88C8VEBA) API key, Hugging
  Face token — both optional, both only for smoother downloads of gated or
  rate-limited repos) are stored on your disk and sent only to the service they
  belong to.

Unified memory is why this works at all on a 16 GB machine: the CPU and GPU
share one pool, so a quantised model stays resident with no CPU↔GPU copying —
exactly what MLX is built to exploit.

---

## Model vs model — visual arena

Head-to-head comparisons on original, royalty-free scenes. Explore interactive side-by-side comparison sliders on **[diffusionbear.com/benchmark.html](https://www.diffusionbear.com/benchmark.html)** or view our video write-up at [ouinche.com](https://www.ouinche.com/mlx-diffusion-yet-another-open-source-image-generator-on-apple-silicon/).


---

## Everything else the app does

- **Model browser & downloader** — search Hugging Face and Civitai side by side,
  classify, download with progress/speed/cancel, direct URLs, and optionally
  convert raw single-file Civitai SDXL `.safetensors` checkpoints out-of-process
  when external `torch`/`diffusers` packages are present (every bundled engine
  runs 100% PyTorch-free on MLX and `safetensors`).
- **Reference images** — FLUX.2/FLUX.3 take up to 10 (`Image 1`, `Image 2`, …),
  Bria FIBO 2.0 takes up to 4 (`image_1` to `image_4`), Z-Image, Krea 2, Qwen
  and FLUX.1 take one; studio-canvas backdrops on the box-capable engines.
- **LoRA manager** — drag a `.safetensors` in or import from Civitai; FLUX.1,
  FLUX.2, FLUX.3, Z-Image, Krea 2 and SDXL (rank-concat stacking) formats;
  trigger words become clickable chips; a LoRA that does not fit the selected
  model is refused *before* generating, with the mismatch named.
- **Prompt enhancer** — local Qwen2.5 0.5B/1.5B rewrites your prompt per engine
  family, prose or JSON, with editable system prompts and trigger-word
  preservation.
- **Custom samplers** — from a local `.py` file or a URL; protected built-in
  default; a broken custom sampler falls back to the engine default, never to a
  failed generation.
- **Queue & recovery** — one generation at a time behind a FIFO worker; a job
  interrupted by a crash or quit is recoverable and re-queueable.
- **Gallery** — tags, search, favourites, one-click Use-as-Reference, 2×/4×
  Lanczos upscaling, Civitai-compliant PNG metadata (with a stealth mode that
  omits it all).
- **Interface** in English, French, German, Italian, Spanish, Simplified
  Chinese, Japanese, Portuguese and Korean.
- **⚖ Licences tab** — the licence of every bundled package and every supported
  model, in one place.

---

## Remote browser & API keys (`Chrome`, `Firefox`, `Opera`, `Safari`, AI Agents)

Because DiffusionBear serves its entire React studio interface from the Mac (`127.0.0.1:8001`), another computer, iPhone, or iPad needs **only one web browser**:

1. **Host Mac (`Parameters → API`):** create a managed key (`dbk_...`) per device, click **Use in this browser**, flip **API mode → ON**, and expose loopback over HTTPS with `tailscale serve --bg --https=443 http://127.0.0.1:8001` (or launch the CLI with `--host 0.0.0.0` on a private LAN).
2. **Remote browser (`Chrome`, `Firefox`, `Opera`, `Safari`):** open `https://<mac>.ts.net` (or `http://<LAN-IP>:8001`), go to **Parameters → API → Backend connection**, **leave `Backend URL` empty** (already talking to the Mac serving the page), paste the key, click **Test → Save connection**.
   - **Chrome:** prefer `https://` via Tailscale Serve (avoids Mixed Content) and allow *Local network access* on LAN HTTP. Keys live per profile in `localStorage`.
   - **Firefox:** open `https://<mac>.ts.net` or a numeric IP (`192.168.x.x` / `100.x.y.z`) if DNS over HTTPS is on, and allow site storage.
   - **Opera:** turn **OFF Opera's built-in VPN** for LAN/Tailscale URLs so traffic stays inside your private network.
   - **Safari (macOS / iOS / iPadOS):** pair Tailscale HTTPS (`https://<mac>.ts.net`) or trusted LAN HTTP (`http://<LAN-IP>:8001`), leave **Backend URL empty**, paste your `dbk_...` key, and tap **Test → Save connection** so all calls and `<img>` thumbnails stay strictly same-origin under *Prevent Cross-Site Tracking*. On iPhone/iPad, **Share → Add to Home Screen** creates a full-screen web app with its own `localStorage` (paste the key once inside the Home Screen app too).
3. **Avoid locking yourself out (`security.json` recovery):** managed `dbk_...` keys are revealed once at creation (stored as SHA-256 hashes) and saved per browser profile (`DiffusionBear.app`, Chrome, Brave, Safari do not share `localStorage`). Save your key in a password manager and click **Use in this browser** before flipping **API mode → ON**. If locked out on your own Mac, edit `~/Library/Application Support/DiffusionBear/data/security.json` (or `backend/data/security.json`) and set `"enabled": false`—your keys stay intact and loopback opens immediately.
4. **AI agents:** send `Authorization: Bearer $DB_KEY` or `X-API-Key: $DB_KEY` on `/api/*` (`GET /api/version` stays public; `GET` image/upload routes accept `?token=` for `<img>` tags). Full guide: [`docs/parameters-api-guide.md`](docs/parameters-api-guide.md) · [`docs/agentic-api-reference.md`](docs/agentic-api-reference.md).

---

## For developers

Everything above is the app; this section is the minimum needed to clone, build
and run it from source.

**Requirements:** an Apple Silicon Mac, macOS 26.2 or later, Python 3.13 (pure
MLX runtime — PyTorch, `diffusers`, and `accelerate` are completely removed
across every bundled engine, including SDXL), and Node.js 18 or later. Git
and Homebrew are only needed to install those prerequisites.

**Clone the repository:**

```bash
git clone https://github.com/OuincheWinch/DiffusionBear.git
cd DiffusionBear
```

**Build the standalone app** (recommended — downloads a self-contained Python
runtime, so no local Python setup is needed). Run the two scripts in order; the
result lands in `packaging/dist/DiffusionBear.app`:

```bash
packaging/build_runtime_venv.sh   # runtime: Python 3.13 + mflux + transformers/tokenizers
INSTALL_APP=1 packaging/build_app.sh   # assembles, signs and installs to /Applications
```

Omit `INSTALL_APP=1` to build without installing. The scripts audit their own
output and fail on absolute build paths, unrelocatable interpreters, weights or
credential-shaped files. Always use the build script's install step — a hand
`cp -R` of a signed bundle is known to silently skip files.

**Run from a checkout** (development):

```bash
python3.13 -m venv venv
./venv/bin/python -m pip install -r backend/requirements.txt
./venv/bin/python -m pip install --no-deps -r backend/requirements-nodeps.txt
D="$(mktemp -d)" && cp -R packaging/vendor/mflux-src "$D/" && ./venv/bin/python -m pip install --no-deps --force-reinstall "$D/mflux-src"; rm -rf "$D"
cd frontend && npm install && cd ..
./run.sh        # backend on 8001, Vite on 5174, opens the UI
```

Dev mode is `./dev-backend.sh` plus `cd frontend && npm run dev`; lint and build
are `npm run lint` and `npm run build`. Backend tests run with
`./venv/bin/python -m unittest discover` from `backend/`. Ports 8001/5174 are
deliberate — do not "fix" them.

**Public-source export.** `packaging/export_public.sh` builds a filtered tree
(internal docs, tests, scratch excluded), audits it for local paths,
credential-shaped files and weights, and refuses to ship anything unsafe;
`--push` publishes to the public repository.

---

## Documentation

- **[Agentic REST API Guide](https://www.diffusionbear.com/agentic-api.html)** & [site/assets/agentic-api-reference.md](site/assets/agentic-api-reference.md) — local endpoints, agent system prompt, and automation examples
- [CHANGELOG.md](CHANGELOG.md) — what changed in each release
- [RELEASE_NOTES.md](RELEASE_NOTES.md) — release notes and the Gatekeeper guide
- [CONTRIBUTING.md](CONTRIBUTING.md) — how to build, verify and propose a change
- [SECURITY.md](SECURITY.md) — security policy and vulnerability reporting
- [PRIVACY.md](PRIVACY.md) — when the app goes online and what stays local
- [USER_GUIDE.md](USER_GUIDE.md) — full usage guide
- [docs/VirtualBox.md](docs/VirtualBox.md) — How Virtual Boxes work
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) — open-source dependency and model licence notices

---

## Author and AI-assisted development

DiffusionBear is written and maintained by **[Ouinche](https://www.ouinche.com)**
(<https://github.com/OuincheWinch>). Follow on X:
**[x.com/DiffusionBear](https://x.com/DiffusionBear)**.

DiffusionBear is **heavily co-coded by AI** — LLM do a
large share of the writing. That is a deliberate choice, not an accident.

## Security

Please **do not open a public issue** for a security problem. Use GitHub's
[private vulnerability report](https://github.com/OuincheWinch/DiffusionBear/security/advisories/new),
or email the author -hello -at- -DiffusionBear-.-com- with the subject `[DiffusionBear security]`.
Scope is in [SECURITY.md](SECURITY.md). Network behaviour and local data are in [PRIVACY.md](PRIVACY.md).

## Third-party credits

Native MLX MOSS-Image support is adapted from the experimental
[`vanch007/mflux`](https://github.com/vanch007/mflux) branch
`feature/moss-image-sft` (commit `b1fe66cc62109f374fe7d713411c589d3810d0c0`),
which builds on [Filip Strand's mflux](https://github.com/filipstrand/mflux)
under the MIT licence. The vendored adaptation lives in `backend/moss_mlx/`.

MOSS-Image weights are by the OpenMOSS Team (Apache-2.0), converted to MLX Q4 by
mflux-community. The Qwen 3.5 text encoder retains the Qwen licence. Full terms
are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

- **Built with Qwen**: The local prompt enhancement (Qwen2.5) and Qwen-Image generation capabilities are powered by models from Alibaba Cloud.

### Third-Party Models

- **MOSS & KROMOSS**: Powered by models from the OpenMOSS Team, licensed under Apache 2.0.
- **FIBO**: Generation and editing powered by Bria AI's FIBO models, licensed under CC BY-NC 4.0.
- **DreamLite**: The instant 0.39B mode is powered by carlofkl's DreamLite models, licensed under CC BY-NC 4.0.
- **Anima**: The 2B anime engine is powered by CircleStone Labs' Anima, under the CircleStone Labs Non-Commercial License (weights non-commercial, outputs commercial; derivative of NVIDIA Cosmos-Predict2 under the NVIDIA Open Model License).
- **Ming-Image**: Typography, UI, and layout-heavy design generation is powered by inclusionAI's Ming-Image 0.2 Design (24.9B, MIT) via [ming-image-swift](https://github.com/xocialize/ming-image-swift) (MIT).
- **ControlNet branches** (optional, downloaded once on first ControlNet use): FLUX.1-schnell ControlNet-Canny and Z-Image-Turbo ControlNet-Union-2.1 q4 packs from mflux-community (Apache-2.0), and the Qwen-Image-2.1-Turbo ControlNet-Union q4 pack (Qwen Research License, like the Qwen-Image 2.1 base).

## License & Commercial Model Rights

[MIT](LICENSE) — Copyright © 2026 **[Ouinche](https://www.ouinche.com)**. This
covers DiffusionBear's source code and the packaged PyTorch-free MLX runtime
(`torch`, `torchvision`, `torchaudio`, `diffusers`, and `accelerate` are not
bundled or redistributed). Model weights are never bundled — they are downloaded
directly from Hugging Face or [Civitai](https://civitai.com/?ref_code=88C8VEBA)
under their upstream terms. The in-app **⚖ Licences** tab shows full package
and model terms offline; always verify trademark, copyright, and likeness rights
before commercial sale:

| Model | Upstream Licence | Sell generated images? | Weights / API / key conditions |
|---|---|---|---|
| **FLUX.2-klein 4B**, **FLUX.1-schnell** | `Apache-2.0` | **Yes** | Permissive commercial use & weights (preserve licence/attribution notices) |
| **MOSS-Image SFT 5B**, **MOSS-Image Base 5B**, **Kromoss 5B** | `Apache-2.0` | **Yes** | Permissive commercial use & weights |
| **Ming-Image 0.2 Design 24.9B** | `MIT` | **Yes** | Permissive commercial use & weights (MIT) |
| **Z-Image Turbo 6B**, **Qwen-Image Edit** | `Apache-2.0` | **Yes** | Permissive commercial use & weights |
| **Qwen2.5-1.5B / 0.5B Instruct** *(prompt enhancers)* | `Apache-2.0` | **Yes** | Permissive commercial use & weights |
| **RealVisXL V5.0** & **Lightning** | `OpenRAIL++` | **Yes** | Commercial outputs & weights allowed under OpenRAIL++ ethical-use terms |
| **Juggernaut XL Lightning** | `CreativeML OpenRAIL-M` | **Yes** | Outputs allowed; paid API / hosted commercial deployment requires RunDiffusion licence |
| **FLUX.2-klein 9B** | `FLUX Non-Commercial License` | **Yes** *(Outputs)* | Section 2(d) allows commercial **Outputs** (filters + AI disclosure required); **Weights** remain Non-Commercial |
| **FLUX.3-klein 5.6B** | `FLUX Kommunity License v1.0` | **Conditional** | Free commercial use/outputs under **US$5,000,000/yr** combined gross revenue (filters + AI disclosure) |
| **Krea 2 Turbo 13B** | `Krea 2 Community License` | **Conditional** | Free commercial use/outputs under **US$1,000,000/yr** trailing 12-month revenue incl. affiliates |
| **Qwen-Image 2.1** & **Turbo** | `Qwen Research License` | **No** | Research & evaluation only; commercial deployment requires separate Qwen licence. The ControlNet-Union branch (alibaba-pai) ships under the same Qwen Research License |
| **ControlNet branches** — FLUX.1-schnell Canny q4 & Z-Image-Turbo Union-2.1 q4 | `Apache-2.0` | **Yes** | Optional branch weights (mflux-community packs derived from the Apache-2.0 schnell / Z-Image-Turbo bases), downloaded once on first ControlNet use. The Qwen-Image-2.1-Turbo ControlNet-Union branch follows the Qwen Research License row above |
| **FIBO 2.0 Edit Turbo**, **Bria RMBG-3.0** | `CC BY-NC 4.0` | **No** | Non-commercial only; commercial licence available from Bria AI |
| **DreamLite Mobile 0.39B** | `CC BY-NC 4.0` | **No** | Non-commercial weights **and** outputs; attribution required |
| **Anima Turbo 2B** | `CircleStone Labs Non-Commercial License` | **Yes** *(Outputs)* | Weights non-commercial (no paid API / commercial hosting); generated images may be sold; derivative of NVIDIA Cosmos-Predict2 (NVIDIA Open Model License) |
| **Juggernaut XI v11** | `CC BY-NC-ND 4.0` | **No** | Strictly non-commercial, no derivatives, no paid API or redistribution without RunDiffusion licence |

*Third-party LoRAs and Civitai checkpoints (e.g., Outlined icons LoRA) carry their own terms shown on the LoRA row in the app.*
