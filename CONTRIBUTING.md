# Contributing to DiffusionBear

Thanks for helping with the public beta! This is a small, personal project — a fully local image-generation studio for Apple Silicon. Everything below is intentionally light.

## Getting started

```bash
# One command: backend (8001) + frontend (5174) + browser
./run.sh
```

Development (2 terminals):

```bash
./dev-backend.sh                       # FastAPI, port 8001, hot reload
cd frontend && npm run dev             # Vite, port 5174, hot reload
```

### Repo venvs
- `venv/` — main backend, **Python 3.10**, deps in `backend/requirements.txt`
- `venv-sdxl/` — SDXL engine, **Python 3.14**, deps in `backend/requirements-sdxl.txt` (mlx_diffuser from GitHub)
- Never run bare `pip` — always `./venv/bin/pip …`

### Verify your changes
```bash
./venv/bin/python -m py_compile backend/main.py backend/generator.py backend/prompt_enhancer.py && ./venv/bin/python -m compileall -q backend
cd frontend && npm run lint && npm run build
```
(The GitHub Actions workflow runs exactly this on every PR.)

## Ground rules

- **Apple Silicon only.** MLX is macOS/Metal exclusive. Don't add non-Metal paths.
- **Ports are fixed:** backend **8001**, frontend **5174**. 8000/5173 belong to other local tools.
- **One generation at a time** — the worker thread + FIFO queue in `state.py`/`generator.py` is intentional. Don't introduce parallel inference.
- **16 GB M1 is the reference machine** — every model and setting must fit 16 GB unified memory and stay fast. Optimizations get benchmarked, not assumed.
- **Releases are version-stamped in one place:** `frontend/src/version.js` and `backend/app_version.py` must stay in sync with `frontend/package.json`.
- **Licences matter** — if you add a dependency or a new model, update the ⚖ Licences tab data in `frontend/src/data/licences.js` (and/or `README.md`).

## A note on this project's DNA

DiffusionBear is **heavily coded by AI** (Gemini, 0xAlpha, Big Pickle) alongside its author [Ouinche](https://www.ouinche.com). AI-assisted patches are expected and welcome — but every change still needs a human review, a clear description, and a benchmark or reproduction when it touches performance.

## Pull requests

1. Keep PRs small and focused.
2. Describe *what* and *why*, plus the benchmark/verification you ran.
3. Don't bump version or rebuild README screenshots unless asked.
4. Don't commit `backend/data/generated/`, tokens, venvs, or model weights (gitignored anyway).

Questions? Open an issue — GitHub Issues are enabled.