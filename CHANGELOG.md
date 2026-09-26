# Changelog

All notable changes to **MLX-Diffusion** are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [SemVer](https://semver.org/).

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