==========================================================================
DiffusionBear — User Guide (English)
==========================================================================
Local image-generation studio optimized for Apple Silicon (MLX & Metal GPU).
Backend: FastAPI (Python)   ·   Frontend: React 19 / Vite   ·   Engines: FAAS / mflux

All commands below use RELATIVE paths and assume you run them from the
project root — the folder that contains run.sh, backend/ and frontend/.
There is no hardcoded machine path, so the repo works from any location.


==========================================================================
1. REQUIREMENTS
==========================================================================
- Mac Apple Silicon (M1–M4, 16 GB+ Unified Memory recommended)
- macOS Ventura/Sonoma/Sequoia with Metal
- Python 3.10 (main engine venv  `venv/`)  and Python 3.14 (SDXL engine `venv-sdxl/`)
- Node.js 18+ (React/Vite frontend)
- First run downloads model weights (~2–3 GB) into the Hugging Face cache;
  allow ~20 min.

Ports (do not change — 8000 / 5173 are reserved by other local tools):
- Backend  : 8001   (FastAPI, Swagger at http://localhost:8001/docs)
- Frontend : 5174   (React/Vite studio at http://localhost:5174)


==========================================================================
2. QUICK START (RECOMMENDED)
==========================================================================
Full install + launch, from the project root (the folder containing run.sh,
backend/ and frontend/):

Step 0 - get the repo ( download it fron github )

    git clone https://github.com/OuincheWinch/DiffusionBear.git
    cd DiffusionBear

Step 1 — create the two Python virtual environments (both are required:
the main engine venv `venv/` and the isolated SDXL engine `venv-sdxl/`):

    python3 -m venv venv
    python3 -m venv venv-sdxl

Step 2 — install the Python dependencies into each venv:

    ./venv/bin/python -m pip install -r backend/requirements.txt
    ./venv-sdxl/bin/python -m pip install -r backend/requirements-sdxl.txt

Step 3 — install the frontend dependencies:

    cd frontend && npm install && cd ..

Step 4 — launch everything with one command:

    ./run.sh

run.sh will:
1. free stuck processes on ports 8001 / 5174 if any,
2. start the FastAPI backend (`venv/`) with caffeinate (keeps the Mac awake
   during long renders),
3. wait for the backend to be ready, then start the Vite frontend,
4. open http://localhost:5174 in your default browser,
5. stop backend + frontend + SDXL engine cleanly on Ctrl+C.

Your first generation downloads the model weights (~2–3 GB, allow ~20 min)
into the Hugging Face cache — afterwards everything runs fully offline.

Suggested first prompt (works with FLUX.2-klein 4B, Z-Image Turbo 6B or
Krea 2 Turbo 13B):

    "A mischievous baby otter wearing a tiny yellow developer helmet, sitting
     in front of a futuristic glowing computer setup. The glowing computer
     screen clearly displays the words "HELLO WORLD" in vibrant neon text.
     Warm studio lighting, shallow depth of field, 8k resolution, cinematic
     photorealism."

In-context reference tip: appending [Image 1] to the prompt conditions the
generation on a reference image. You MUST load that image into the reference
tray first, and ONLY FLUX.2-klein 4B accepts image input — Z-Image, Krea 2
and SDXL refuse it with:
    Cannot read "HELLOWORLD.png" (this model does not support image input)
If you see that error, switch to FLUX.2-klein 4B or remove the [Image N] tag.


==========================================================================
3. DEVELOPMENT MODE (2 TERMINALS)
==========================================================================
Terminal 1 — Backend (hot reload):
    ./dev-backend.sh
    # equivalent: cd backend && ../venv/bin/uvicorn main:app --reload --port 8001

Terminal 2 — Frontend (hot reload):
    cd frontend && npm run dev

Note: the backend code is 100% live-reloaded on save. Settings are persisted
in backend/data/settings.json (gitignored — never committed).

Production build of the frontend:
    cd frontend && npm run build        # outputs to frontend/dist


==========================================================================
4. STOPPING SERVICES
==========================================================================
Graceful: press Ctrl+C in each terminal (or in run.sh's terminal).

Kill-everything one-liner (stuck daemons):
    pkill -f "uvicorn main:app" ; pkill -f sdxl_engine.py ; pkill -f vite

Free the ports directly:
    lsof -ti :8001,5174 | xargs kill -9

SDXL engine behavior: sdxl_engine.py runs as an isolated subprocess in
venv-sdxl and kills itself after 5 min idle. To reclaim ~7–11 GB memory
immediately:  pkill -f sdxl_engine.py


==========================================================================
5. UNINSTALL (FULL REMOVAL)
==========================================================================
Ensure your terminal is open inside the project folder you wish to remove:

    cd /path/to/your/project-folder
    cd .. && rm -rf DiffusionBear

Purge the leftover caches so nothing lingers on the machine:

    # purge pip cache (clears wheels and downloaded python packages)
    python3 -m pip cache purge

    # clear npm global cache
    npm cache clean --force


==========================================================================
6. SUPPORTED MODELS (ALL QUANTIZED FOR 16 GB UNIFIED MEMORY)
==========================================================================
1. FLUX.2-klein 4B  (main engine — Black Forest Labs DiT)
   - id flux2-klein-4b  ·  ideal 4 steps  ·  guidance 1.0 (guidance-distilled,
     NO negative prompt)
   - ~90 s @ 768x768  ·  ~120–140 s @ 1024x1024 (M1)
   - In-context multi-reference conditioning (1–10 images, referenced in the
     prompt as "Image 1", "Image 2", …)
   - Exact #HEX color matching + on-screen palette insertion
   - Optional PiD VAE super-resolution decode (advanced settings)
   - Multi-LoRA FLUX.2 (.safetensors), slider swap in 0s

2. Juggernaut XL Lightning  (distilled SDXL, ultra-fast)
   - id juggernaut-xl-lightning  (RunDiffusion)
   - 4 steps  ·  euler_trailing  ·  guidance 1.0  ·  ~15–20 s @ 1024x1024 with TAESD
   - ⚡ Fast TAESD VAE decode (~0.5 s) or native VAE
   - Multi-LoRA Kohya/CivitAI with rank-concat stacking, samplers:
     euler_trailing (default), dpmpp_2m_karras, euler_a_substep, euler_a,
     euler, ddim  ·  full negative-prompt support

3. Krea 2 Turbo 13B  (photorealistic)
   - id krea2-turbo (local krea2-turbo-q4)
   - WITHOUT distill LoRA : MUST run 8 steps (4 steps stay blurry)
   - WITH Krea2-Turbo-Distill-4step LoRA : runs in 4 steps (~140 s vs ~250 s)
   - max 512x512 recommended (512x768 portrait on 16 GB)
   - Q4-quantized Qwen3-VL text encoder (7.5 GB → ~1.9 GB)

4. Z-Image Turbo 6B  (fast, large formats)
   - id z-image-turbo  (filipstrand)
   - ~40 s @ 1024x1024  ·  great for 1280x720 16:9 sketches

Swapping models automatically unloads incompatible LoRAs to avoid crashes.


==========================================================================
7. PROMPT ENHANCER (
==========================================================================
Local 4-bit LLM (Qwen2.5-0.5B-Instruct via mlx-lm) — fully offline, ~0.5–1 s,
<350 MB RAM, no cloud call.

- Model-guided rewriting: rules adapt to each engine's text encoder
  (T5 for FLUX.2, dual-CLIP for SDXL, Qwen3-VL for Krea, DiT for Z-Image).
- LoRA trigger preservation: active LoRA trigger words are detected from
  backend/data/loras.json, kept verbatim, and re-inserted deterministically
  if the LLM drops one.
- Editable system prompts: Settings → Prompt Enhancer lets you view and
  customize each engine's guidance in two modes:
  - Text mode — natural-language rewriting rules;
  - JSON mode — the full instruction contract (engine guidance + schema
    structure + fill-in guidelines), applied verbatim when you override it.

The ✨ Enhance button sits right of the prompt field; it replaces the prompt
with an enriched, engine-adapted description in ~1 s.


==========================================================================
8. LoRAs & CIVITAI IMPORT
==========================================================================
- Add LoRA: drag & drop a .safetensors onto the form, or use "+ Add LoRA…"
  (only compatible architectures are offered).
- Civitai import: paste any Civitai URL or numeric model ID; live progress
  bar with speed and cancel (✕). Checkpoint files >2 GB and incompatible
  architectures (SD1.5 / Flux.1) are rejected upfront.
- Installed LoRAs Hub drawer: browse every installed LoRA across SDXL /
  FLUX.2 / Krea / Z-Image, one-click "Switch to {Architecture}", delete 🗑️.
- Multi-LoRA with rank-concat stacking (SDXL) and in-context multi-LoRA
  (FLUX.2). Trigger words live in data/loras.json.

Model install from an already-downloaded copy (no re-download):
  In the installer, "📁 Local…" lets you register a model tree already on
  disk — a folder on your disk OR a repo in the Hugging Face cache
  (~/.cache/huggingface, models--org--name, auto-resolved to its latest
  snapshot). Nothing is copied: the model loads directly from that path, the
  installer marks it "installed", and RAM/disk usage reflect it. Uninstalling
  only unlinks the pointer — your files stay on disk.


==========================================================================
9. STUDIO & GALLERY
==========================================================================
- Split-screen studio: generation form (left) + interactive canvas (right).
- Upscaling: ✨ AI Neural 2x (SeedVR2 latent 1-step), ⚡ Fast 2x / ⚡ Fast 4x
  (Lanczos, 0.18 s).
- Variants: 🎲 randomize, +1 seed, +1024 batch seed.
- Gallery (Browser tab): lazy-loaded thumbnails, date/tag/text search,
  🖼️ "Use as Reference" re-injects any image into the multi-reference tray.
- Civitai-compliant metadata: every PNG embeds prompt / negative / steps /
  sampler / seed / CFG / checkpoint + AutoV2 hashes / LoRAs with version IDs
  in tEXt + EXIF, generator "DiffusionBear" / artist "www.ouinche.com".
  Fully readable by Civitai's upload parser.


==========================================================================
10. TROUBLESHOOTING
==========================================================================
| Symptom                                  | Fix                                                     |
|------------------------------------------|---------------------------------------------------------|
| Port 8001 / 5174 already in use          | lsof -ti :8001,5174 \| xargs kill -9                    |
| MemoryError on LoRA load (HTML/corrupt)  | Delete via UI 🗑️ or DELETE /api/loras/{name}            |
| Civitai 401 Unauthorized                 | Add token: UI API Key field or backend/data/civitai_token.txt |
| Civitai "EXIF not found"                 | python backend/scripts/retroactive_civitai_metadata.py --all |
| Sluggish / swap with both engines loaded | pkill -f sdxl_engine.py                                 |
| Generation fails at step 0 (Metal FE)    | Run outside sandboxes; ensure Metal shader cache access |
| First gen extremely slow                 | First run downloads ~2–3 GB weights (~20 min)           |

==========================================================================
LICENSE
==========================================================================
DiffusionBear is released under the MIT License (see LICENSE file in the repo
root). This license covers the source code only — it does NOT cover the model
weights (FLUX.2-klein is Black Forest Labs Non-Commercial; other checkpoints
carry their own terms) or third-party vendored content.
==========================================================================
