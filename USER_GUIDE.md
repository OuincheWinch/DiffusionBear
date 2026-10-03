# MLX-DIFFUSION — User Guide

Local generative playground for Apple Silicon running on **MLX** and **mflux**.
Supports **FLUX.2-klein 4B**, **Juggernaut XL Lightning (SDXL)**, **Krea 2 Turbo**, and **Z-Image Turbo**.

---

## 1. Requirements

- **Mac Apple Silicon** (M1, M2, M3, M4 — 16 GB+ Unified Memory recommended)
- **macOS Sonoma / Sequoia** with Apple Metal
- **Python 3.10** (for FLUX/Krea/Z-Image in `venv/`) and **Python 3.14** (for SDXL in `venv-sdxl/`)
- **Node.js 18+** (for React/Vite frontend)
- Sufficient disk space for model weights (cached in `~/.cache/huggingface` and `backend/data/models/`)

---

## 2. Starting the Application

### 🚀 Recommended: 1-Command Unified Launch (`run-mlx`)
From any terminal, simply run:
```bash
run-mlx
```
*(Or directly `./run.sh` from the repository root).*

This unified script automatically:
1. Checks and frees ports **8001** and **5174** if previously stuck.
2. Starts the **FastAPI backend** with `caffeinate` (preventing macOS from sleeping during heavy renders).
3. Waits for the backend to initialize, then starts the **Vite frontend**.
4. Opens **http://localhost:5174** in your default browser.
5. Handles `Ctrl + C` gracefully to stop backend, frontend, and SDXL engines at once.

---

### Alternative: Manual 2-Terminal Launch
If you prefer running backend and frontend in separate terminals:

**Terminal 1 — Backend (FastAPI API on Port 8001):**
```bash
cd /Volumes/Externe/IA/MLX-DIFFUSION/backend
../venv/bin/uvicorn main:app --reload --port 8001
```

**Terminal 2 — Frontend (React / Vite on Port 5174):**
```bash
cd /Volumes/Externe/IA/MLX-DIFFUSION/frontend
npm run dev
```

Then navigate to:
> **http://localhost:5174**

⚠️ **Important Port Rules:**
- Frontend runs on port **5174** (configured in `vite.config.js`). Port **5173** is reserved for Align-n-GIF.
- Backend runs on port **8001**. Port **8000** is reserved for Align-n-GIF.
- Never change these ports to avoid conflict with other local services.

---

## 3. Stopping & Killing the App in Shell (Tips & Cheat Sheet)

When shutting down or restarting the application, lingering background processes or stuck ports (`Address already in use`) can occur. Here are the essential shell commands:

### A. Standard Graceful Shutdown
In the active terminal windows running Uvicorn and Vite:
- Press `Ctrl + C`.

---

### B. The "Kill Everything" 1-Liner (Recommended)
If terminal sessions were closed, background tasks are stuck, or the port is busy, run this 1-liner to terminate backend, frontend, SDXL engine, and caffeinate:

```bash
pkill -f "uvicorn main:app" ; pkill -f sdxl_engine.py ; pkill -f "vite" ; pkill -f "caffeinate.*uvicorn"
```

---

### C. Freeing Ports 8001 and 5174 Directly
If Uvicorn throws `[Errno 48] Address already in use` or Vite switches to port 5175:

```bash
# Kill whatever is listening on backend port 8001
lsof -ti :8001 | xargs kill -9

# Kill whatever is listening on frontend port 5174
lsof -ti :5174 | xargs kill -9
```

---

### D. Finding What is Running (Inspection)
To inspect active processes without killing them immediately:

```bash
# Check processes listening on ports
lsof -i :8001
lsof -i :5174

# Search active MLX and diffusion processes
pgrep -fl uvicorn
pgrep -fl sdxl_engine
pgrep -fl python
```

---

### E. SDXL Subprocess Daemon Behavior
- The SDXL engine (`sdxl_engine.py`) runs as an isolated subprocess in `venv-sdxl`.
- It loads when generating with Juggernaut XL and has an automatic **5-minute idle watchdog**: it terminates itself if no new SDXL generations are queued within 5 minutes.
- If you need to immediately free GPU unified memory (reclaiming ~7–11 GB RAM), kill it explicitly:
  ```bash
  pkill -f sdxl_engine.py
  ```

---

## 4. Key Features & How to Use

### Multi-Model Engine
Select your model in the **Generate** tab:
1. **FLUX.2-klein 4B**: High visual fidelity flow-matching model (4 steps default).
2. **Juggernaut XL Lightning (SDXL)**: Ultra-fast native 4-step distilled photorealistic model (`euler_trailing`, guidance 1.0).
3. **Krea 2 Turbo**: 8-step default or 4-step accelerated with the official Krea 2 4-step distillation LoRA.
4. **Z-Image Turbo**: 4-step fast generation.

*Note: Swapping models automatically unloads incompatible LoRAs to prevent engine corruption.*

---

### ✨ Magic Prompt Enhancer
Next to the Prompt input, click **✨ Enhance**:
- Uses an embedded, local 4-bit LLM (`Qwen2.5-0.5B-Instruct`, ~350 MB RAM, ~1s latency).
- Adapts prompt structure to the selected model (FLUX natural language vs SDXL tag/token style).
- **Guaranteed LoRA trigger word preservation**: Automatically detects active LoRA trigger words and keeps them intact in the enhanced prompt.

---

### Civitai LoRA Downloader & Installed Hub
Download LoRAs directly from Civitai via URL or Model ID:
1. Expand **Civitai Import** in the LoRAs section.
2. Paste any Civitai LoRA URL (e.g. `https://civitai.red/models/582432/wagyu-beef-style-sdxl?modelVersionId=654467` or `582432`).
3. If the model requires Civitai login, click **Civitai API Key** and paste your Civitai API token (stored in `backend/data/civitai_token.txt`).
4. Click **Import**:
   - Downloads run asynchronously with **live progress bar, speed (Mo/s), and cancel button (✕)**.
   - Non-LoRA files (checkpoints >2GB) and incompatible architectures (SD 1.5, Flux.1) are rejected upfront.
5. **Installed LoRAs Hub Drawer**:
   - Click the **Installed LoRAs Hub** button to view all installed LoRAs across every architecture (`SDXL`, `FLUX.2`, `Krea 2`, `Z-Image`).
   - Includes 1-click **"Switch to {Architecture}"** button to immediately swap your active generator to match the LoRA.
   - Delete unwanted LoRAs anytime using the 🗑️ icon.

---

### SDXL Fast VAE & Acceleration
Under the SDXL model settings:
- **Native VAE (JIT Accelerated ~8s)**: Bit-for-bit full-precision VAE decoder with dynamic latents tiling.
- **⚡ Ultra-Fast TAESD (~0.5s)**: Pure MLX tiny autoencoder decoding in 500ms with corrected dynamic range and zero haze.

---

### Civitai-Compliant Image Metadata
All generated images embed full generation metadata into standard PNG `tEXt` chunks and EXIF:
- Prompt, Negative prompt, Steps, Sampler, Seed, Guidance/CFG scale.
- Checkpoint name and canonical AutoV2 hashes.
- LoRAs list with exact version IDs and hashes.
- Generator / Software: `"DiffusionBear"`, Artist: `"www.ouinche.com"`.
- 100% compatible with Civitai upload drag-and-drop parsing.

---

## 5. Gallery & Result Canvas Actions

- **Full-Text & Tag Search**: Filter previous generations in the **Browser** tab.
- **Reuse Parameters**: Click to reload prompt, model, sampler, and settings back into the form.
- **Copy Seed**: Reproduce exact variations.
- **Upscaling**:
  - **High-Fidelity Lanczos (2x / 4x)**: Instant bicubic/lanczos crisp scaling.
  - **✨ AI Neural 2x**: Neural latent upscaling via SeedVR2.

---

## 6. Troubleshooting

| Symptom | Cause | Solution |
|---|---|---|
| Port 8001 / 5174 already in use | Previous server did not exit cleanly | Run `lsof -ti :8001 \| xargs kill -9` and `lsof -ti :5174 \| xargs kill -9` |
| `MemoryError` or daemon crash on LoRA load | File is HTML or corrupted | Delete via UI 🗑️ or run `DELETE /api/loras/{name}` |
| Civitai download returns 401 Unauthorized | Model requires login | Add Civitai token via the UI API Key field or `backend/data/civitai_token.txt` |
| Civitai says "EXIF not found" | Non-standard characters in old images | Run retroactive patch: `python backend/scripts/retroactive_civitai_metadata.py --all` |
| System feels sluggish / high swap | Both SDXL and FLUX resident | Run `pkill -f sdxl_engine.py` to reclaim memory |
| Generation fails at Step 0 | Metal FE compiler permissions | Ensure backend is launched outside restrictive sandboxes with access to Darwin cache |

---

## 7. Useful Shell Aliases (Configured in `~/.zshrc`)

To manage DiffusionBear effortlessly from any terminal:

```bash
# 🚀 1-command launch: frees ports, starts backend & frontend, opens browser, traps Ctrl+C
alias run-mlx='/YOUR/PATH/TO/MLX-DIFFUSION/run.sh'

# 🛑 Kill all MLX-DIFFUSION processes (backend, SDXL engine, Vite, caffeinate)
alias kill-mlx='pkill -f "uvicorn main:app" ; pkill -f sdxl_engine.py ; pkill -f "vite" ; pkill -f "caffeinate.*uvicorn"'

# 🧹 Force-free MLX ports (8001 & 5174)
alias free-mlx='lsof -ti :8001,5174 | xargs kill -9 2>/dev/null'

# 🔍 Check MLX status
alias status-mlx='lsof -i :8001 -i :5174 ; pgrep -fl "sdxl_engine|uvicorn"'
```
