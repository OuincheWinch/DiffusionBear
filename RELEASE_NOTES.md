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
