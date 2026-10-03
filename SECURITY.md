# Security Policy

## Privacy posture

DiffusionBear is a **local-first** application:

- Generation, prompt enhancement, and LoRA management run entirely on-device. No telemetry, no analytics, no network calls during generation.
- Network is only touched when **you** explicitly trigger it: downloading model weights/LoRAs (Hugging Face, Civitai), or uploading/exporting an image.
- Tokens (Civitai API key, Hugging Face token) are stored on your disk in `backend/data/` and sent only to the service they belong to. They are gitignored and never leave your machine otherwise.
- Images embed Civitai-compliant prompt/seed/model metadata in PNG `tEXt` + EXIF by design (that is the point of the gallery), so exported images carry that data with them.

## Reporting a vulnerability

Please **do not open a public GitHub issue** for security problems. Report privately instead:

- Open a [private vulnerability report](https://github.com/OuincheWinch/MLX-Diffusion/security/advisories/new) (GitHub's "Report a vulnerability" flow), or
- Email the repo author with the subject `[DiffusionBear security]`.

You should get an acknowledgement within 3 working days. Treat PoCs as embargoed until the issue is fixed or declined.

## Scope

In scope: the FastAPI backend (`backend/`), the React frontend (`frontend/`), and the build/run scripts. Out of scope: third-party packages, ML conversion repos, and model weights (report those to their own maintainers), and any data in your local `backend/data/` that is already private to your machine by design.

## Supported versions
| Version | Supported |
|---|---|
| 0.1.x (beta) | ✅ patches via GitHub Issues/releases |
| Older | ❌ upgrade |