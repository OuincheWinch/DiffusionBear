# Engineering notes

Operational notes for this repository.

> This started life as `AGENTS.md` in the `MLX-DIFFUSION OpenCode` checkout, where it was
> untracked and one reset away from being lost. It now lives here, in version control.
> `AGENTS.md` is deliberately gitignored in this repo (`# ---- Personal docs (kept local
> only) ----`), so it is not tracked under that name; local tooling that looks for
> `AGENTS.md` will still find a copy kept alongside it, untracked as the policy intends. Read this before changing memory, packaging or
download behaviour — most of what is here was learned the expensive way, and several
plausible-sounding claims in it have already been disproven once.

## Overview

DiffusionBear: a fully local image-generation studio for Apple Silicon. FastAPI backend +
React 19 / Vite SPA, generating with mflux on MLX.

- `backend/` — FastAPI (`main.py`); generation in `generator.py`; downloads in
  `routers/downloads.py`; HF discovery in `hf_browse.py`. Images + JSON sidecars in
  `data/generated/`; LoRA registry in `data/loras.json`.
- `frontend/` — SPA. `src/App.jsx` owns tab state, `src/api.js` is the only HTTP client,
  `src/i18n/` holds a nine-language catalogue.
- `packaging/` — the standalone `.app` (see Gotchas).
- `tasks/todo.md` — what is in flight. It has been stale before; check the code before
  believing a "not implemented" line in it.

## Commands

Backend — **always** `./venv/bin/python -m uvicorn main:app`, from `backend/`:

```bash
./venv/bin/python -m uvicorn main:app --port 8001
```

Never `./venv/bin/uvicorn`: the console-script shebangs are stale (they point at a
leftover no-space `.../MLX-DIFFUSION/venv` copy whose mflux lacks the community-layout
text-encoder prefix fix), and macOS truncates a shebang at the first space so they cannot
be repointed at the real spaced path.

```bash
cd backend && ../venv/bin/python -m unittest discover -q     # 464 tests
cd backend && ../venv/bin/python -m unittest test_i18n        # catalogue guard
```

Frontend:

```bash
cd frontend && npm run dev        # vite dev server, port 5174
npm run lint                      # oxlint
npm run build                     # production build
npm run i18n:build                # regenerate lang/{es,zh,ja,pt,ko}.js from lang/_parts/*.json
```

Standalone app — `packaging/build_runtime_venv.sh` → `build_sdxl_venv.sh` →
`build_app.sh`, then `cp -R packaging/dist/DiffusionBear.app /Applications/`.

`build_app.sh` fails the build on any credential-shaped file, `.safetensors`, or stray
large file inside the bundle, and that check is the only thing standing between a
mistake and a token in a signed app.

## Where state actually lives

This is the single most confusing thing about working on this project, and it has cost
real time.

- The **running app** writes to `~/Library/Application Support/DiffusionBear/data` —
  **and that is a symlink into `MLX-DIFFUSION OpenCode/backend/data`**, not a separate
  store. `Path.resolve()` canonicalises it, so a backend started with
  `MLX_DIFFUSION_ASSET_DIR` pointed at Application Support will report a path under the
  other tree. That is correct. Do not "fix" it.
- There is a second checkout, `MLX-DIFFUSION OpenCode`, at an older commit. Its gallery,
  `settings.json`, LoRA registry and model store are the live ones. It is a **separate git
  repo** — committing there does not reach private DEV.
- `MLX_DIFFUSION_ASSET_DIR` moves only the model store; `MLX_DIFFUSION_DATA_DIR` moves the
  gallery and settings. Old `MLX_DIFFUSION_*` names still win over `DIFFUSIONBEAR_*`.
- **Only one process may hold port 8001.** A stray manual backend will not fail loudly; the
  app shows a "Port 8001 is already in use" dialog and serves nothing. Check with
  `lsof -nP -iTCP:8001 -sTCP:LISTEN` before concluding the app is broken.

## SDXL (Juggernaut XL Lightning) — second engine

Runs via `venv-sdxl/` as a subprocess; `backend/sdxl_engine.py` is the entry point. The
engine runtime does not need **torch**, but it DOES need **transformers** for the SDXL
tokenizer. The main runtime DOES need torch: mflux's weight loader imports it on the
inference path.

## MLX memory findings (2026-10-02)

Four things cost most of a session. Three of my diagnoses were wrong before the real cause
was measured, so these are recorded to stop the next one repeating them.

- **mflux ships Krea 2's text encoder as bf16 while labelling it `quantization_level: 4`.**
  `mflux-community/krea-2-turbo-mflux-q4` does contain a `text_encoder/` directory, and its
  index *does* claim quantisation — but it has 399 keys and **zero `.scales`**, and the four
  shards total 7672 MB where a real q4 encoder is ~2 GB. It is bf16; the label is wrong. The
  cause is `skip_quantization=True` on that component in the weight definition, so
  `mflux-save` skips it too. There is **no pre-quantised Krea 2 encoder to download** — every
  user on every quantisation pays the 7.5 GB bf16 load. We now persist our own: the first
  build quantises once and writes `<model>-te-q4-g64-<sha10(path)>.safetensors` (2158 MB) into
  `ASSET_DIR/models`; later builds read it back at **2.26 GB** with no bf16 and no quantise
  pass. Load order is the trap: `Krea2()` loads bf16 *before it returns*, so the fix has to
  drop `text_encoder` from the `models` dict inside `WeightApplier._set_weights` **before**
  constructing the pipeline, not after.
- **`mx.get_peak_memory()` is not a memory metric. Do not reason from it.** It reported
  **24.29 GB on a 16 GB machine**, and it saturates at the same value across unrelated
  configs. Two of my "peak" tables were therefore meaningless. The only trustworthy signals
  are *did the process survive* and *is the output a real image* (count distinct colours — a
  blank PNG here had exactly 1).
- **Metal's maximum single buffer on this machine is 9,534,832,640 bytes (8.88 GiB).** A hard
  device limit; no MLX setting or env var moves it. Separate from, and much smaller than,
  total memory.
- **Converting an MLX array to numpy aborts the entire process.** `np.array(mx_array)` makes
  numpy pull the buffer protocol, which calls `array::eval()` from inside a C callback.
  pybind11's exception translator only wraps Python→C calls, so a Metal failure there escapes
  to `std::terminate` → `SIGABRT` / "Abort trap: 6", and the whole backend dies with no error
  in the UI. Measured in a subprocess: `mx.eval()` in Python scope raises a catchable
  `RuntimeError`; `np.array()` un-evaluated aborts. That is why `generator.py` wraps
  `np.array` **process-wide** and evaluates MLX input first — the pre-existing
  `ImageUtil._to_numpy` patch did not cover the real call site, and we did not find that site
  until the guard printed its stack.
- **1024×688 plus a reference image is a dead end on this 16 GB M1 — user decision
  2026-10-02, do not re-litigate.** The failing frame is the reference VAE encode:
  `generate_image → _prepare_latents → LatentCreator.encode_image → VAEUtil.encode`. Note
  the reference is pre-scaled to the output dimensions by `latent_creator.py:78`, so this is
  *not* a 4.2 MP encode — an earlier note claiming that was wrong. 512×768 works. VAE decode
  tiling (`TilingConfig`, assign `pipe.tiling_config` after construction) is on by default
  and is safe for Krea 2; leave `vae_encode_tiled` at its `True` default.
- **The wired budget is auto-derived now**, not a constant: `min(68% of unified memory,
  Apple's recommended working set)`, cached per process — 10.88 GB here, ~43 GB on 64 GB.
  An explicit number is still honoured as a deliberate pin and `0` still means unbounded.
  `mx.set_wired_limit` is a hint, not a cap: it reported 14.96 GB of peak against a 10.88 GB
  budget. A previous explicit-pin experiment was reverted because its benefit did not
  reproduce under an interleaved A/B and it bought nothing measurable.

## Gotchas

- **NEVER capture the screen or audio.** No `screencapture`, no `arecord`, no ffmpeg device
  capture, no headless `--screenshot`/`--print-to-pdf` of a browser window, no window/app
  automation that reads pixels back. The user does not permit it. It was requested once and
  explicitly refused; treat it as a hard boundary, not a default to be tested. Verify visual
  output by reading the file's data (dimensions, bytes, parsed JSON) and by telling the user
  where the artifact is so *they* can open it.
- **Never delete or deduplicate gallery images.** The user wants to see every image the app
  has ever generated.
- **Do not push to `upstream`.** `origin` is the private DEV repo; `upstream` is the public
  one and stays untouched unless explicitly asked.
- **Standalone app** (`packaging/`) — one-click `.app`, ad-hoc signed, no Developer ID.
  Load-bearing details, each learned the hard way:
  - The interpreter is **python-build-standalone**, not a copy of `venv/`. The repo venv is
    neither self-contained nor relocatable.
  - A pbs venv's `bin/python3` must stay a **relative** symlink. Rewriting it as a copy makes
    `sys._base_executable` point into the venv, which has no stdlib, and the engine dies with
    "No module named 'encodings'". Sign only real Mach-O files, never symlinks.
  - `Resources/` layout is dictated by the backend:
    `{backend, venv, venv-sdxl, python, frontend/dist}`.
  - `LSMinimumSystemVersion` is **15.0**: MLX ≥ 0.29.2 (0.32.1 here) calls a Metal API absent
    on Sonoma.
  - **Never set `PYTHONHOME`** when spawning the engine — it leaks into every grandchild.
  - The launcher's readiness probe must **not** require a 200 from a `HEAD`: FastAPI registers
    only GET for a plain `@app.get`.
  - `Bundle.main.resourceURL` is unreliable here; derive paths from
    `Bundle.main.executableURL` by walking up two levels.
  - Never exclude only `data/models` when copying the backend: `data/lora_files` is 425 MB
    **and `data/hf_token.txt` is a real credential**. The whole `data/` tree is excluded.
- **No router library.** Tabs are a `useState` string in `App.jsx` and every view stays
  mounted, hidden with `display: none`. Moving sections between tabs therefore does not
  unmount anything — watch fetch timing, and note `Gallery` suppresses its own fetches when
  `activeTab !== "browser"`.
- **i18n: French is the reference language**, and every key must exist in all nine.
  `npm run i18n:build` validates that each `_parts/<group>.json` group name matches the key
  namespace; `test_i18n.py` fails on missing languages, and on untranslated Latin in CJK
  strings. Do not add UI strings without a `_parts` entry in all five generated languages.
- **Model download progress must use `transform: scaleX()`**, never an animated `width` —
  `backend/test_progress_animation.py` fails the build otherwise.
- **Architecture and quantisation exist only as substrings of a Hugging Face repo id.**
  There is no structured field in the HF API and no quantisation field in this app's model
  registry (mflux infers it implicitly). Any filter on either has to parse names.
- **Generation is a `model_id == ...` dispatch onto a fixed set of mflux classes.** A
  downloaded repo becomes runnable by pointing the existing `settings.model_paths` override at
  its directory (it feeds `local_arg` → `model_path=`), not by editing generation code.
  Anything outside that set — every upscaler, Illustrious, Pony — is downloadable but not
  executable, and `upscale.py` has no neural upscaler at all (PIL Lanczos + UnsharpMask).
- **One generation at a time** — worker thread + lock in `generator.py`.
- FLUX.2-klein 4B is guidance-distilled: **no negative prompt** support.
- Model pipeline is cached in memory keyed by (quantization, LoRA paths, variant) and released
  by a watchdog `MFLUX_IDLE_KILL_S=300`. **Eviction costs nothing measurable** (measured 2026-09-28
  across immediate / 60 s / 180 s gaps, no trend). The old "~20-40 s cold reload" note was stale
  and sent sessions chasing a phantom cost.
- **MLX is lazy, so `_get_pipeline()` return time is not load time.** To measure a load, time
  rebuild **plus** `mx.eval(pipe.transformer.parameters())`.
- SDXL on M1 16 GB: UNet must run `quantize_unet=4` (~5x faster than fp16); 8-bit is ~7x
  **slower**, never enable it. Check system load before benchmarking — load > 5 doubles gen times.
- When asked to improve or optimise something, do **not** patch the obvious spot: research
  alternatives (other frameworks, engines, algorithms — CoreML vs MLX, driving external tools'
  APIs), benchmark the cheap ones, and propose before or alongside implementing.