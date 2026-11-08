# Security Policy

## Privacy posture

DiffusionBear is a **local-first** application:

- Generation, prompt enhancement, captioning and LoRA management run entirely on-device. There is no telemetry, analytics, crash reporting, background update check or cloud account.
- The app does not contact the network at launch or in the background. Requests happen only in response to an action you start: a catalogue search, a model/LoRA download, a licence refresh, a sampler install, or a generation/enhancement for which you ticked **Allow required downloads**. The full list is in [PRIVACY.md](PRIVACY.md).
- Tokens (Civitai API key, Hugging Face token) are stored on your disk with restricted permissions and sent only to the service they belong to. They are gitignored and never leave your machine otherwise.
- Images embed Civitai-compliant prompt/seed/model metadata in PNG `tEXt` + EXIF by default, so exported images carry that data with them. Stealth mode saves them without metadata.

Read [PRIVACY.md](PRIVACY.md) for the exact network paths, local storage locations and deletion steps.

## Reporting a vulnerability

Please **do not open a public GitHub issue** for security problems. Report privately instead:

- Open a [private vulnerability report](https://github.com/OuincheWinch/DiffusionBear/security/advisories/new) (GitHub's "Report a vulnerability" flow), or
- Email the repo author with the subject `[DiffusionBear security]`.

You should get an acknowledgement within 3 working days. Treat PoCs as embargoed until the issue is fixed or declined.

## Scope

In scope: the FastAPI backend (`backend/`), the React frontend (`frontend/`), and the build/run scripts. Out of scope: third-party packages, ML conversion repos, and model weights (report those to their own maintainers), and any data in your local `backend/data/` that is already private to your machine by design.

## Supported versions
| Version | Supported |
|---|---|
| 0.1.x (beta) | ✅ patches via GitHub Issues/releases |
| Older | ❌ upgrade |