# DiffusionBear Release Notes

This document contains release notes for each published version of **DiffusionBear**, as well as the standard first-launch guide required for macOS Gatekeeper.

---

## 🍎 Important Note for macOS Users (Gatekeeper / First Launch)

Because DiffusionBear is an independent open-source project built with local ad-hoc code signing (without an expensive paid Apple Developer ID notarization certificate), macOS Gatekeeper protects your Mac by halting the initial double-click with this security alert:

<p align="center">
  <img src="docs/screenshots/gatekeeper-blocked.png" alt="macOS Gatekeeper Alert — DiffusionBear Not Opened" width="340">
</p>

> **“DiffusionBear” Not Opened**  
> *Apple could not verify “DiffusionBear” is free of malware that may harm your Mac or compromise your privacy.*

This is standard macOS security behavior for free open-source software. To authorize DiffusionBear:

1. **Move to Applications:** Drag `DiffusionBear.app` into your **/Applications** folder (do not launch it directly from inside the mounted DMG or `~/Downloads`).
2. **Authorize it (choose one of the two methods):**
   * **Method A (Terminal 1-Liner — Fastest):**  
     Open **Terminal.app** and run:
     ```bash
     xattr -cr /Applications/DiffusionBear.app
     ```
     *(Or: `sudo xattr -rd com.apple.quarantine /Applications/DiffusionBear.app`)*  
     Then simply double-click `DiffusionBear.app` — it opens immediately!
   * **Method B (System Settings — GUI):**  
     1. Click **Done** on the alert dialog.
     2. Open **System Settings > Privacy & Security**.
     3. Scroll down to the **Security** section: you will see *“DiffusionBear” was blocked from use because it is not from an identified developer.*
     4. Click **Open Anyway**, enter your Mac password / Touch ID, and confirm **Open**.

Once authorized, the warning will **never return**, and DiffusionBear will launch normally on every future click.

---

## Release Notes Template (For Future Releases)

```markdown
# DiffusionBear vX.Y.Z

**Release Date:** YYYY-MM-DD  
**Target OS:** macOS 26.2+ (Apple Silicon M1 / M2 / M3 / M4 / Ultra)  
**Package:** `DiffusionBear-X.Y.Z-arm64.zip` / `DiffusionBear-X.Y.Z.dmg`

### 🍎 First Launch on macOS (Gatekeeper)
Before opening the app for the first time, move `DiffusionBear.app` to `/Applications` and run:
`xattr -cr /Applications/DiffusionBear.app`
*(See the top of RELEASE_NOTES.md for full GUI and terminal instructions.)*

### ✨ Highlights
- Key feature 1
- Key feature 2

### 🚀 What's New
- Feature details

### 🛠️ Fixes & Improvements
- Fix 1
- Fix 2

### 📦 Checksums
- `shasum -a 256 DiffusionBear-X.Y.Z-arm64.zip`: `<sha256>`
```

---

## [0.4.3] — beta — 2026-10-29

### ✨ What's New
- **Browser page revamp:** navigate pages with `←` / `→` and jump directly to a page number.
- **Memory management tweaks:** large generations are checked against the machine's safe megapixel ceiling before the engine starts, preventing Metal out-of-memory crashes.
- **Minor bug fixes.**

### 📦 Package
- `DiffusionBear-0.4.3-arm64.zip`
- SHA-256: `d456806513034b6d7fdfe9301f84ac9b8e21c9a0594421ee1161bf124ac4f88c`

### 📦 Checksums
- `shasum -a 256 DiffusionBear-0.4.3-arm64.zip`: `d456806513034b6d7fdfe9301f84ac9b8e21c9a0594421ee1161bf124ac4f88c`

---

## [0.4.2] — beta — 2026-10-27

### ⚠️ Post-Mortem & Incident Report: v0.4.2 Packaging & Dependency Optimization (Full Transparency)

We believe in complete engineering transparency with our users. Here is what happened with the v0.4.2 release packaging iterations:

#### 1. What Happened (The Initial FIBO Bug)
- The initial v0.4.2 build published on 2026-10-26 (465.6 MB, SHA-256 `cde8d5...`, ~20 early downloads) worked for FLUX.2, SDXL, Krea 2, and Z-Image Turbo.
- However, when attempting to generate with the newly introduced **FIBO 2.0 Edit Turbo**, the app immediately failed with `Generation failed: ModuleNotFoundError: No module named 'mlx_vlm'`.
- **Root Cause**: During local development, FIBO was tested against a local `./venv-fibo/` virtualenv. To keep the app bundle lean, this folder was not copied into the `.app`. Instead, FIBO runs on the main bundled runtime `Resources/venv`. However, `mlx-vlm` was omitted from `packaging/build_runtime_venv.sh`, so users on a clean install lacked the module.

#### 2. The Second Mistake: The 139 MB Dependency Bloat
- When adding `mlx-vlm>=0.7.10` to the build script, running a standard `pip install` pulled the package's default `setup.py` dependencies.
- `mlx-vlm` declared heavy external training dependencies: `pyarrow` (122 MB), `scipy` (80 MB), `pandas` (41 MB), and `matplotlib` (20 MB) — totaling **263 MB of unused code**.
- This produced an inflated temporary package of **604.6 MB** (SHA-256 `9a22c6...`).

#### 3. The Third Mistake: Stripping cv2 (`ModuleNotFoundError: No module named 'cv2'`)
- In an urgent attempt to purge the bloat, `mlx-vlm` was installed with `--no-deps` and all dependencies were stripped, assuming Pillow and MLX handled all image operations.
- **The fatal flaw**: While `pyarrow`, `scipy`, `pandas`, and `matplotlib` were indeed dead weight, `cv2` is a **critical runtime dependency**:
  1. `mflux.models.z_image` imports `controlnet_util.py`, which eagerly executes `import cv2`. Consequently, any generation with **Z-Image Turbo** failed instantly with `ModuleNotFoundError: No module named 'cv2'`.
  2. `backend/fill.py:744` uses `cv2.seamlessClone` for Poisson blending during Studio inpainting fill compositing.
- Standard `opencv-python` is 119 MB uncompressed because it bundles GUI Qt frameworks and FFmpeg codecs.

#### 4. The Final Engineering Solution: `opencv-python-headless` (502 MB)
- `packaging/build_runtime_venv.sh` was updated to explicitly bundle `opencv-python-headless>=4.9.0` alongside `mlx-vlm --no-deps`.
- `opencv-python-headless` is only 48 MB uncompressed, contains zero GUI or video decoder bloat, and restores 100% full runtime support for OpenCV operations (`seamlessClone`, controlnet filters, Z-Image Turbo).
- Added automated guards in `backend/test_fibo_integration.py` (`test_cv2_available_in_runtime` and `test_z_image_turbo_importable_without_cv2_error`) and verified all 113 unit and integration tests pass.
- Added `cv2` and `mflux` to the build script relocation rehearsal gate to ensure no un-relocatable dynamic library ever slips through.
- The clean, fully operational package is **502.0 MB** (103 MB lighter than the bloated 605 MB build):
  - **SHA-256**: `e30878491df578486c5c5594d965b8a535e88561fcaa66cedf9c593eaa77fb80`

#### 5. Audit Table of Build Iterations
| Build Iteration | Size | SHA-256 Checksum | Status |
| :--- | :--- | :--- | :--- |
| **v0.4.2 Verified Release (Current)** | **502.0 MB** | `e30878491df578486c5c5594d965b8a535e88561fcaa66cedf9c593eaa77fb80` | ✅ **Fully operational across all models** |
| *v0.4.2 Missing cv2 Build* | *456.0 MB* | *`62794333b85a164112d065f8cdf24a6310964a22953acb47aa1e46a29440220e`* | ❌ *Missing `cv2` (Z-Image Turbo & fill fail)* |
| *v0.4.2 Bloated Build (Temporary)* | *604.6 MB* | *`9a22c6bd35b1d49fd858830dfdd4ffaf096f83f977e9d52c856358fcffc7ccc4`* | ⚠️ *Functional but carries 263 MB pyarrow/scipy bloat* |
| *v0.4.2 Initial Release (~20 downloads)* | *465.6 MB* | *`cde8d562fbbfaafa91de1acf8e80e36de2d8330b2824498569ec1c09b2a73308`* | ❌ *Missing `mlx_vlm` (FIBO generation fails)* |

---

### 🍎 First Launch on macOS (Gatekeeper)
Before opening the app for the first time, move `DiffusionBear.app` to `/Applications` and run:
`xattr -cr /Applications/DiffusionBear.app`
*(See the top of RELEASE_NOTES.md for full GUI and terminal instructions.)*

### ✨ Highlights
- **Full Support for FIBO 2.0 Edit Turbo (MLX 4-bit):** First-ever 4-bit affine quantized port of Bria AI's state-of-the-art instruction editing model running natively on Apple Silicon via MLX ([`Ouinche/FIBO2-Edit-Turbo-MLX-4bit`](https://huggingface.co/Ouinche/FIBO2-Edit-Turbo-MLX-4bit)).
- **Text-Guided Single Reference Edit:** High-fidelity single reference editing with natural language prompts in only 4 denoising steps (~85s on baseline M1).
- **Studio Masked Edit (Precision Inpainting):** Interactive brush painting directly on reference image backdrop in DiffusionBear Studio. Zero variance outside masked area with automated 1.5px Gaussian edge blending and unmasked composite retention.
- **Multiple References (Multimodal In-Context Editing):** Combine up to 4 reference images (e.g., character portrait + specific dress or object) grounded through the integrated 4-bit Qwen3-VL vision-language encoder.
- **3-Stage Sequential Memory Management:** Lazy component lifecycle (Qwen3-VL → DiT Transformer → FLUX.2 VAE) keeping resident VRAM strictly below 6.8 GB on 16 GB Apple Silicon.
- **Visual Announcement & Benchmark Page:** Interactive showcase, side-by-side comparisons, and telemetry report at [diffusionbear.com/fibo.html](https://www.diffusionbear.com/fibo.html).

### 📦 Checksums
- `shasum -a 256 DiffusionBear-0.4.2-arm64.zip`: `e30878491df578486c5c5594d965b8a535e88561fcaa66cedf9c593eaa77fb80`

---

## [0.4.1] — beta — 2026-10-24

### 🍎 First Launch on macOS (Gatekeeper)

DiffusionBear is ad-hoc signed, so Gatekeeper blocks the first double-click of a downloaded copy — one-time, expected. Move `DiffusionBear.app` to `/Applications`, then either run `xattr -cr /Applications/DiffusionBear.app` in Terminal, or use System Settings → Privacy & Security → **Open Anyway**. *(Full guide at the top of this file.)*

### 🌟 Release Highlights: SDXL Performance Recovery & Unified Memory Optimization

![DiffusionBear v0.4.1 beta](docs/screenshots/0.4.1/041_announce.png)

DiffusionBear v0.4.1 resolves SDXL memory thrashing on Apple Silicon Macs, elevates generation throughput, and introduces our complete empirical 187-generation multi-model benchmark suite.

- **SDXL Memory Thrashing Root-Cause Resolution (+51.9% Suite Acceleration):**
  - **The Issue:** Persistent daemon caching left up to 7.8 GB of stale Metal allocations resident between runs. When clamped against an artificial 6.5 GB limit, subsequent runs suffered from severe Metal buffer evictions and paging stalls (ballooning latencies up to 237s/image).
  - **The Fix:** Systematic cache purging (`gc.collect()` and `mx.clear_cache()`) inside execution `finally:` blocks and daemon loops, combined with removing artificial clamps in favour of dynamic system working-set allocation.
  - **Results:** Global SDXL suite generation time plunged from **4,569.5s (1h 16m 09s)** down to **2,195.9s (36m 35s)**: **+51.9% faster across the board, saving over 39 minutes of compute**.
  - `Juggernaut XL Lightning`: **4.25x faster (+76.5%)**, down from 423.2s to **99.6s** (strictly 9.15s–9.30s per image).
  - `RealVisXL V5.0`: **3.63x faster (+72.5%)**, down from 1,709.9s to **470.8s**.
  - `Juggernaut XI`: **2.15x faster (+53.4%)**, down from 988.3s to **460.8s**.
  - `Pony`: **+19.6% faster**, down from 1,448.0s to **1,164.7s**.

- **Bit-Identical Visual Fidelity Recovery (&infin; dB PSNR):**
  - Distilled SDXL models (4-step Lightning, 8-step Hyper-SD) require trailing timesteps down to $t=0$. DiffusionBear now automatically promotes them to `euler_trailing` sampling.
  - Verification: `Juggernaut XL Lightning` achieved **&infin; dB PSNR (0.00 / 255 mean pixel difference)** against our September golden baseline — restoring 100% bit-for-bit mathematical visual parity.

- **High-Performance TAESD VAE Decoding:**
  - Integrated standalone `taesdxl` autoencoder for ultra-lightweight, sub-second latent decode (~0.45s), removing VAE decoding bottlenecks.

- **Complete 187-Run Empirical Benchmark Arena:**
  - Dual-benchmark architecture preserving legacy data while introducing all 17 modern configurations across 4 engine families (FLUX, Krea, DiT, SDXL) with an interactive side-by-side comparison slider on both `site/` and `docs/`.

- **Hardened Daemon Lifecycle & Broken Pipe Resilience:**
  - Robust IPC handling protecting against dead socket handles, broken pipes, and ensuring zero orphaned processes on exit.

- **Custom Model Dimensions & Reactive Dimension Tags:**
  - The Parameters tab now exposes and displays exact default dimensions (width and height) for each model. Users can customize and persist model-specific defaults.
  - Active custom dimensions dynamically reflect in the Generate tab with a clear `"Custom"` tag indicator in the model picker and size selector.

- **High-Contrast Dark Mode Form Affordance:**
  - Explicit `color-scheme: dark;` configuration and high-contrast WebKit stepper filters restore crisp visibility to up/down spinners (`input[type="number"]`), select arrows, and navigation chevrons.

### 📦 Checksums

- `shasum -a 256 DiffusionBear-0.4.1-arm64.zip`: `097ccacd46cd3b7707389dadcbd865ea5c4a7b3c5b71f58d5c409d60880ae08c`

---

## [0.4.0] — beta — 2026-10-20

### 🍎 First Launch on macOS (Gatekeeper)

DiffusionBear is ad-hoc signed, so Gatekeeper blocks the first double-click of a downloaded copy — one-time, expected. Move `DiffusionBear.app` to `/Applications`, then either run `xattr -cr /Applications/DiffusionBear.app` in Terminal, or use System Settings → Privacy & Security → **Open Anyway**. *(Full guide at the top of this file.)*

### 🌟 Release Highlights

- **Expanded 12+ Model Lineup Across 8 Families:**
  - **SDXL (and its derivatives):** `Juggernaut XL Lightning`, `RealVisXL V5.0 Lightning`, `RealVisXL V5.0 (Hyper-SD)`, `Juggernaut XI v11`, plus custom single-file SDXL, Pony, and Illustrious `.safetensors` checkpoints (auto-converted up to 50 steps) with adjustable **CLIP Skip**.
  - **FLUX.1:** `FLUX.1-schnell` (4-step flow-matching with full FLUX.1 LoRA support and fast-VAE decode).
  - **FLUX.2:** `FLUX.2-klein 4B` and `FLUX.2-klein 9B` (up to 10 reference images, LoRAs, and generative fill).
  - **FLUX.3:** `FLUX.3-klein 5.6B` (16-step flow-matching distilled engine, up to 10 reference images, and interactive **Virtual Boxes** spatial layout).
  - **MOSS-Image:** `MOSS-Image SFT 5B` (and `Base 5B`) with interactive **Virtual Boxes**, reference image, LoRAs, MLX q8-g64 text encoder, and automatic **4-step (`⚡ DMD 4` darask0) and 8-step (`⚡ DMD 8` lodestones) DMD step-distillation** adapters (~1–3 min vs. 30 min stock), downloaded on first use with silent fallback to the stock trajectory when offline.
  - **Qwen-Image:** `Qwen-Image-2.1` and `Qwen-Image-Turbo-2.1` (8-step distilled variant).
  - **Krea 2:** `Krea 2 Turbo 13B` (with persistent Q4 text-encoder disk cache eliminating the 7.5 GB bf16 spike).
  - **Z-Image:** `Z-Image-Turbo 6B` (fast 6–8 step generation with sharp geometry).
- **Interactive Virtual Boxes (Studio Canvas HUD):** Draw, drag, resize, and label color-coded spatial boxes (`[ymin, xmin, ymax, xmax]`) directly on the Studio Canvas for **FLUX.3-klein 5.6B** and **MOSS-Image SFT 5B** to place subjects and poster typography without ControlNet.
- **Smart 16 GB Unified Memory Management:**
  - **Auto Wired-Memory Budget & 5-Minute Idle Watchdogs:** Every engine releases its pipeline after 5 idle minutes (tunable, `0` = never release), and the GPU wired limit auto-tunes to your Mac (`min(68% of unified memory, Apple's recommended working set)`, ≈ 10.9 GB on a 16 GB machine). Parameters → Engine & GPU shows live Metal usage, resident pipelines and idle countdowns.
  - **Auto-Unloading Prompt Enhancer:** Reclaims **1.2 GB** of RAM immediately before image generation begins.
  - **512px Cosine-Blended VAE Tiling:** Reduces peak 1024×1024 decode memory by **4.5 GB** with zero visible tile seams.
  - **MLX q8-g64 MOSS Text Encoder & Krea 2 Q4 Text-Encoder Cache:** The app quantises the MOSS encoder once and persists its own ~2.2 GB copy instead of loading 7.5 GB of bf16; Krea 2's encoder is quantised once and reused on later sessions.
- **Gallery Favorites & Privacy Card Blur:** Filter your best images with one-click **☆ Favorites**, and toggle **Blurred** privacy overlays on individual gallery cards (click to reveal).
- **User-Declared Custom Samplers:** Declare your own sampler from a local Python file (`.py`) or a URL — the UI stores it, it appears with a ★ in the sampler menu, and the built-in default sampler stays protected and re-extracts itself if missing or broken.
- **DiffusionBee & Draw Things Local Library Import:** Browse and import models and LoRAs already downloaded by **DiffusionBee** (`.tdict`) and **Draw Things** with a strict read-only guarantee.

### 🛠️ Fixes & Reliability

- **Cached-LoRA crash fixed:** re-running a generation with the same LoRA set hit `RuntimeError: LoRA target layers.… is unavailable` on the cached pipeline; LoRA wrappers are now unwrapped before re-apply.

### 💻 System Requirements

- **Mac:** Apple Silicon only. Intel Macs are not supported.
- **macOS:** 26.2 or later — derived from the minimum-OS load commands of the shipped binaries (MLX, libJACCL); the older 15.0 floor was stale and understated it eleven majors.
- **Memory:** 16 GB unified memory recommended; model weights are not bundled and download on first use into `~/Library/Application Support/DiffusionBear`.

### 📦 Checksums

- `shasum -a 256 DiffusionBear-0.4.0-arm64.zip`: `1e7446aa4848d66055640fcc655c40f434f8a1a9c501902fd70de9bed546407c`

---

## [0.3.9] — beta — 2026-10-18

### 🌟 Release Highlights

- **FLUX.3-klein 5.6B Engine & Flow-Matching Sampler:** Integrated Black Forest Labs' next-gen 16-step flow-matching distillation pipeline (`mlx-community/flux3-image-klein-4bit`) with spatial bounding box prompt conditioning (`[ymin, xmin, ymax, xmax]`), multi-reference image composition (up to 10 images), and strict step bounding (4–24 steps).
- **Studio Canvas & VirtualBox HUD:** Interactive bounding box overlay on the canvas with draggable crosshair positioning, normalized 0–1000 coordinate alignment, real-time prompt sync, and intuitive Clear/Reset actions.
- **Custom SDXL & Pony Checkpoint Conversion:** Automatic out-of-process conversion of Civitai single-file SDXL checkpoints into native diffusers layout, custom models family dispatch (Pony / Illustrious non-distilled SDXL support with up to 50 steps), and 1-click registration of unrecognised models on disk.
- **External Local Library Discovery:** Native binary `.tdict` parser and SQLite reader to browse and convert external LoRAs and models from **DiffusionBee** and **Draw Things** with a strict read-only guarantee.
- **Persistent Multi-Engine Download Telemetry:** Download progress bars remain active across model tab switches and background operations, backed by real-time speed, ETA, and cancellation controls.
- **Active Generation Header Lock & Color-Coded Indicators:** Header bar permanently displays the running model during active generations even when browsing other engines; model lists feature high-contrast color coding (vibrant green for runnable local models, red for missing/unrunnable checkpoints).
- **Dual Prompt Enhancer:** Direct 2-position toggle (0.5B / 1.5B) with live status indicator and system prompt leakage filtering.
- **Studio Canvas Action Toolbar:** Consolidated 2-row action toolbar with 1-click parameter reuse (`Reuse params`), instant variation, super-resolution upscale (2x / 4x), native file drag/export, and metadata inspector.
- **Nine-Language Localization:** 100% complete catalogue across French (reference), English, German, Italian, Spanish, Chinese, Japanese, Portuguese, and Korean.

### 🛠️ Bug Fixes & Reliability

- **LoRA Dimension & Architecture Guard:** LoRA adapters are validated against model architecture before generation, preventing crashes 90 seconds into inference with mismatched layer errors.
- **Atomic Downloads & Stale Temp Cleanup:** Downloads stream into temp files with SHA-256 validation before replacing target files, preventing corrupted partial models from breaking the registry.
- **SDXL Persistent Engine Subprocess:** SDXL generation persists loaded UNet/VAE weights across generations via a background daemon rather than reloading ~6.5 GB on every run.
- **Metal Memory Stability on 16 GB Macs:** Auto-derived wired memory budget (`min(68% unified memory, Apple recommended working set)` = 10.88 GB limit) preventing SIGABRT buffer overflow.

### 💻 System Requirements

- **Mac:** Apple Silicon only (M1, M2, M3, M4, Pro, Max, Ultra). Intel Macs are not supported.
- **macOS:** macOS 26.2 or later (derived from Mach-O load command minimum OS floor).
- **Memory:** 16 GB unified memory recommended for optimal multi-model performance.
