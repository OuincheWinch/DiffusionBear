# Changelog

All notable changes to **DiffusionBear** are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [SemVer](https://semver.org/).

## [0.3.6] — beta — 2026-10-07

### Fixed
- **Half-downloaded models were reported as INSTALLED.** The installed-status
  probe ignored `.incomplete`/`.part` files under `.cache/` — the exact directory
  `huggingface_hub` stages `local_dir` downloads in — so an unfinished transfer
  read as installed: no Download button, `already_installed` from the API, and
  generation then correctly refusing it. Both probes now count every partial, and
  a killed run's orphaned partial is dropped instead of failing the installer's
  layout check forever (an 18-day-old 13 MB orphan beat every later verification).
- **A failed generation pasted the engine's traceback into the UI.** SDXL daemon
  stderr stays in the diagnostics log; the job error keeps the first actionable
  line and the generation reference. Hugging Face non-checkpoint names are matched
  on token boundaries now, so `exploration`, `floral` and `brave` no longer read
  as `lora`/`vae`.
- **The steps slider promised steps the engine would never run.** A guidance-
  distilled model clamps in the engine (Juggernaut XL Lightning to 12), so a
  20-step request ran 12 while reporting "step 12/20". `/api/models` now exposes
  `max_steps`, the slider tops out there under a "Distilled — max N steps" tag,
  and the progress readout, the ETA and the run agree.

### Changed
- **The one-click preset chips are gone.** They duplicated the size picker and the
  now-capped steps slider, their labels came from the registry untranslated, and
  the row sat between the model picker and Generate.
- SDXL's auto-injected 4-step Lightning LoRA now refuses to stack on top of a
  second distillation adapter (Hyper-SD, LCM) instead of recognising Lightning by
  filename alone.

## [0.3.5] — beta — 2026-10-05

### Fixed
- **Every SDXL model was reported as "Not runnable here".** The table that decides
  which architectures the app can execute covered FLUX.2, Z-Image, Krea and Qwen and
  had no SDXL entry at all — including the registry's own repository, which showed an
  INSTALLED badge beside the claim that it could not run. Four SDXL engines are now
  recognised, and ControlNets, LoRAs, VAEs and upscalers are rejected up front:
  `juggernaut-xl-lightning-4step-controlnet-coreml-6b` matches every required token
  and is not a checkpoint.
- **A failed SDXL load now says what to do** instead of ending in a forty-line
  traceback with `No .safetensors files in .../unet`. Interrupted download,
  single-file checkpoint, empty `unet/` and never-installed each have their own
  message, because they need different fixes.
- **A download was marked installed without being checked.** The task was declared
  successful when the transfer loop finished, without looking at what had been
  written, so an interrupted 9.5 GB shard left an empty `unet/` behind an
  INSTALLED badge.
- **The fast TAESD decode was dead in every shipped build.** It resolved its
  weights from `__file__/data`, which inside the signed bundle is inside the
  signature, so every SDXL render with fast VAE silently fell back to the full VAE.
- **The bundle named the build machine in 88 files** — 82 console-script shebangs,
  both `pyvenv.cfg` homes, six `activate` scripts, a `direct_url.json` and three
  docstrings. Fixed by construction: relative `pyvenv.cfg` homes and `/bin/sh`
  trampolines, so there is no absolute path left to get wrong.
- **23 translations were never actually checked.** The catalogue parser demanded
  exactly two leading spaces while two files indent some entries four and one zero.
  The bound is now exact zero.
- **`Consolas`, a Microsoft font**, was named in two stylesheets and could only ever
  resolve on a Mac with Office installed. Removed.

### Added
- **The app moves port instead of refusing to start.** If something else owns
  8001, it tries 9025, 10049, 11073 … 17217 (ten rungs of +1024). A rung already
  serving a DiffusionBear backend is reused rather than stepped over. Telling our
  backend from someone else's needs the JSON to name DiffusionBear — a 404 from an
  unrelated server looks identical to a healthy answer otherwise.
- **A native-resolution size picker**, replacing a base × ratio control that
  produced sizes no model was trained on.
- `backend/sdxl_layout.py`, `backend/test_font_licensing.py`,
  `backend/test_sdxl_discovery.py`, and a relocation rehearsal that copies the built
  bundle elsewhere and runs it.

### Changed
- The production bundle contains **zero web fonts**, and font licensing is now
  enforced by a default-deny test rather than by inspection.

## [0.3.4] — beta — 2026-10-04

### Fixed
- **The Hugging Face token was saved to one path and read from another.**
  `POST /api/tokens` writes `DATA_DIR/hf_token.txt`; `hf_service.get_hf_token`
  resolved `Path(__file__).parent / "data"`, which inside the app is the signed
  bundle. The token was stored correctly and never read back, so the UI reported
  it unset after every relaunch and gated downloads had nothing to authenticate
  with. Now resolved through `app_settings.DATA_DIR`, matching Civitai.

- **An interrupted download could not be restarted.** The installer rendered its
  actions only when no task had ever existed, so a cancelled or failed download
  left the row with a status and no button. The download button now returns
  whenever nothing is in flight, labelled **Retry**, and `hf_hub_download`
  resumes from the bytes that arrived. Fixed on both the Generate page and the
  Models page, which share the component.

### Added
- **Unrecognised model directories can be adopted.** The storage panel already
  flagged directories nothing referenced; it now proposes which registry model each
  one resembles and can point that model at it, using the same `model_paths`
  override the downloader writes. The directory becomes selectable, stops being
  reported, and nothing is copied — adoption is a pointer, costs no disk, and
  unregistering undoes it.

### Changed
- **The model list is ordered installed-first, then alphabetically** by label, with
  adopted local models sorting by name among the others instead of at the end.
  Sorted in `/api/models` so the Generate dropdown, Models tab and defaults section
  cannot drift apart.

## [0.3.3] — beta — 2026-10-04

### Fixed
- **A fresh install could not finish downloading a model.** Two independent faults,
  both reproduced on a clean machine with an empty store and no Hugging Face cache:

  - `flux2-klein-4b` downloaded `black-forest-labs/FLUX.2-klein-4B` — 22.1 GB of
    upstream weights the app never loads, including a duplicate 7.4 GB checkpoint and
    demo JPEGs — while the pipeline reads `mlx-community/flux2-klein-4b-4bit` at
    4.3 GB. The special case was justified by a comment claiming mflux falls back to
    its own default repo when `model_path` is absent; the code passes `model_path`,
    so the comment was wrong. **17.8 GB saved per install.**
  - The transfer then **stalled rather than failed**. Hugging Face's Xet storage
    backend stopped writing part-way through a 2 GB shard — 575 MB, zero bytes of
    progress, blob lock still held — and raised nothing, so the task sat at 99% for
    ever. Xet is now disabled, which restores the resumable HTTP path, and a stall
    watchdog fails the task after 180s of silence with a message that says what
    happened and that a retry resumes.

  After both: 4.3 GB, 14 files, `installed successfully!` in about 14 minutes,
  detected as installed, no incomplete blobs.

### Note
- The progress bar saturates at 99% by design while files are still moving, because
  the byte counter is clamped to the total and tqdm can over-report across files. The
  file counter in the status line is the honest one.

## [0.3.2] — beta — 2026-10-04

### Fixed
- **Model downloads failed for every mflux model.** The download worker called
  `_hf_repo_cache_dir(repo)`, which lives in `generator.py`, without qualifying it
  or importing it, so the task failed immediately with `name
  '_hf_repo_cache_dir' is not defined`. A fresh install could not obtain any
  model. SDXL downloads were unaffected, which made the failure look
  model-specific rather than total.

  Four tests now cover the download path — including a static check that every
  bare name `routers/downloads.py` calls is defined or imported in that module —
  so the next unqualified cross-module call fails in CI rather than on a click.

## [0.3.1] — beta — 2026-10-03

### Added
- **Civitai as a model source**: browse/search Civitai alongside Hugging Face and download
  checkpoints directly, with token auth and structured `baseModel` classification into the
  SDXL/Flux/Krea buckets.
- **Single-file → diffusers conversion** for Civitai SDXL checkpoints, run out of process so a
  slow conversion cannot stall the API. Normalises CLIP `text_model.*` keys and restores the
  slow tokenizer's `vocab.json`/`merges.txt`, which diffusers loaders otherwise choke on.
- **Register a model from a local folder**, so a converted or hand-assembled diffusers
  directory can be run without touching the model store.
- **Models tab**: detected downloads, runnable-model registration, and honest states for
  models that are present but not runnable.

### Changed
- **New icon and official artwork** — the neon bear/goggles badge is now the app icon, the
  header logo and the favicon, regenerated into `AppIcon.icns` at all required sizes.
- Version is defined once in `backend/app_version.py`; `frontend/src/version.js` and
  `package.json` are generated from it, and a test fails if they drift.
- EXIF `Software`/`Generator` now read `DiffusionBear`. Images written by earlier versions
  keep their original value.

### Fixed
- **Krea 2 text encoder cache**: the community q4 release ships Krea's encoder as bf16 while
  labelling it `quantization_level: 4` (399 keys, zero `.scales`, 7.5 GB). The first build now
  quantises once and persists a ~2.1 GB q4 copy, cutting later loads to 2.26 GB.
- **Wired-memory budget** is auto-derived from unified memory and Apple's recommended working
  set instead of a hardcoded pin, removing a stale 9 GB setting that starved the encoder pass.
- **SDXL conversion failures** now surface the subprocess traceback instead of a bare timeout.
- **Civitai downloads** follow redirects securely and resolve the API token from the settings
  file.

### Known limitations
- Qwen-Image 2.1 needs roughly 5 GB of free memory before it starts. Below that the bf16 VAE
  decode exhausts Metal and the engine dies after writing the image but before reporting
  success, so the job shows as failed while a valid PNG is on disk.

## [0.1.2] — beta — 2026-09-26

### Added
- **Shared model store** via `MLX_DIFFUSION_ASSET_DIR`: several working copies (e.g. a private dev tree beside a public one) can share one set of model weights, SDXL assets and LoRAs instead of re-downloading tens of GB. Generated images, uploads and the LoRA registry stay local to each instance, and deletion is refused for shared SDXL/Krea assets.
- **`./run.sh -s|--silent`**: keeps the launch header and Uvicorn startup output, drops HTTP access logs, and prints one structured line per generation.
- **Generation event log**: each run logs a timestamp, model, prompt, size, steps, seed, batch, guidance, sampler, LoRAs, elapsed time, status and a clickable `file://` link to the output image.
- **128×128 and 256×256** accepted for every model, for quick iteration and pipeline checks. Alignment (16 for mflux engines, 8 for SDXL) and the 2048 maximum are unchanged.
- **Backend test suite** (26 tests) covering dimension rules across every model, image metadata redaction, upscale limits and queue recovery, plus a **supply-chain guard** that fails the build on dangerous code patterns, un-allowlisted outbound hosts, committed credentials, npm lifecycle hooks, unpinned remote dependencies, active git hooks and untrusted workflow triggers.
- `civitai.green` accepted alongside `civitai.com` and `civitai.red`.

### Fixed
- **Civitai downloads could not complete.** The client passed `trust_env` to `requests.get()` (it is a `Session` keyword), called the removed `Response.read()`, rejected valid redirects that carry no peer metadata, and aborted on normal short socket reads. Downloads are now SHA-256 verified and survive redirects and partial reads.
- **Model list reported incomplete installs as ready.** A missing `model_index` component, a stale `.cache` `*.incomplete` file, or an orphaned Hugging Face partial could each mark a model installed. All three are now reported honestly.
- **Generation rejected its own LoRA payload.** The form sent full registry entries while the API validates a strict `{path, scale}` shape; generations failed with `extra_forbidden`. Only the two accepted fields are sent now.
- Local paths are redacted from embedded image metadata.

### Changed
- Version is read from `backend/app_version.py` at launch, so the header and `/api/version` cannot drift.
- CI installs the dependencies the tests actually import, runs on `ubuntu-24.04`, uses node24 actions pinned to commit SHAs, and keeps them current through Dependabot. The repository previously shipped no backend tests, so the CI test step had been passing without running anything.

## [0.1.1] — beta — 2026-09-22
First public beta.

### Added
- **⚖ Licences tab** in the app: per-package + per-model licence tables with repo links, our MIT © Ouinche reminder, and a link to the GitHub repository.
- **Version stamping**: `v0.1.1 (beta)` in the page title and app header; new `/api/version` endpoint; `frontend/src/version.js` and `backend/app_version.py` as the single version sources.
- **Public beta docs**: `CHANGELOG.md`, `CONTRIBUTING.md`, `SECURITY.md`, and a GitHub Actions CI workflow (backend `py_compile`, frontend lint + build).
- README: author credit (Ouinche — www.ouinche.com), AI-assist credit (Gemini, 0xAlpha, Big Pickle), beta disclaimer, refreshed License section.

### Changed
- Prompt Enhancer tightened to a strict output contract (prompt only, hard word cap per engine, no commentary) and tagged *experimental* in the UI.
- Qwen-Image 2.1: demoted to *experimental* — now uses the validated `flow_match_euler_discrete` recipe (quality/portrait presets at 40 steps), refuses >768×768 with a clean error instead of OOMing, and shows a confirmation warning before launch.
- Size caps (`max_pixels`/`max_side`) removed for Krea 2 Turbo and Qwen-Image 2.1 (16 GB caveats documented).
- Header brand moved to consistent `MLX-Diffusion` casing.