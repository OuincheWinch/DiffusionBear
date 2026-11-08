# Privacy

DiffusionBear runs on your Mac. Prompts, images, reference images, masks, LoRAs,
gallery history, settings and tokens stay on your Mac. There is no account, no
cloud sync, no telemetry, no analytics and no crash reporting. Nothing is sent
to the DiffusionBear project. The Hugging Face Hub library's own telemetry is
disabled at startup (`HF_HUB_DISABLE_TELEMETRY=1`).

This document describes the behaviour of **DiffusionBear 0.4.5**. It was last
updated on **6 November 2026**.

## When the app goes online

The app does **not** contact the network at launch, in the background, or on a
timer. It does not run a catalogue search until you open **Models → Browse**.
There is no update check and no “phone home” request. Network is used only in
response to an action you start, and it goes directly from your Mac to the
service involved.

| Service | When | What is sent |
| --- | --- | --- |
| `huggingface.co` and its file storage | You open **Models → Browse** with Hugging Face selected, press Search or Retry, refresh model licences, download a model/LoRA, or tick **Allow required downloads** and start a generation/enhancement that needs an auxiliary file | Your search text, filters, sort order and limit; repository ids and file names; your Hugging Face token if one is configured. The token is sent to `huggingface.co`, not to the file-storage host. |
| `civitai.com` and its file storage | You open **Models → Browse** with Civitai selected, press Search or Retry, match a local LoRA to a Civitai page, refresh LoRA licences, or download a model/LoRA from Civitai | Your search text, model/version ids and download ids; your Civitai API key if one is configured. The key is sent only to `civitai.com`; it is removed when Civitai redirects the download to third-party storage. Local LoRA files are not uploaded. |
| A URL you paste | You use the LoRA downloader or install a custom sampler from a URL | An HTTP request to exactly the URL you supplied. That host has its own privacy policy. |
| Links you click | You click a model, licence, repository or website link | The link opens in your normal browser. DiffusionBear does not proxy, log or read that page. |

The only hosts contacted by the app itself are Hugging Face, Civitai, and a URL
you explicitly paste. Download redirects may reach the storage/CDN hosts chosen
by Hugging Face or Civitai; those requests go to the signed download URL, not
through the DiffusionBear project.

### Required support files

A generation or enhancement may need an auxiliary file that is not installed
yet. **By default the app does not fetch it.** The action fails with a message
telling you what is missing. You can then either install the file explicitly
from the Models/Settings UI, or tick **Allow required downloads** in
**Generate → Advanced Settings** before starting that run.

When the box is ticked, the app may fetch one of these from Hugging Face:

- the shared Qwen tokenizer needed by a FLUX model;
- the Qwen-Image 2.1 DMD2 2-step LoRA;
- the MOSS-Image DMD adapter or its MLX text-encoder bundle;
- the prompt-enhancer model selected in Settings.

Model weights themselves still require an explicit download in the Models tab.
Model weights and LoRAs have their own licences. See the in-app **⚖ Licences**
tab before commercial use.

## What is never sent

DiffusionBear never sends any of the following to the project or to a third
party as part of normal operation:

- your prompts or negative prompts;
- generated images, reference images or masks;
- LoRA files or model weights you created;
- your gallery, favourites, tags, queue or generation history;
- your settings, file paths, seeds or hardware information;
- analytics, telemetry, crash reports or usage statistics.

Searching a model catalogue sends the search text and filters, not your prompt.
Generating an image runs entirely on your Mac. Prompt enhancement, captioning
and LoRA compatibility checks also run locally.

## Tokens and secrets

- A Hugging Face token or Civitai API key is optional. It is needed only for
  gated/private downloads or authenticated Civitai access.
- Tokens are stored locally in the app data directory, with file permissions
  restricted to your user (`0600`), and the data directory uses `0700`.
- The app never logs a token and never shows it again after saving; the UI shows
  only a masked value.
- A Hugging Face token is sent only to `huggingface.co`. A Civitai key is sent
  only to `civitai.com` and is stripped from redirects to file storage.
- If you have not saved a Hugging Face token in DiffusionBear, the app may use a
  token from `HF_TOKEN`/`HUGGINGFACE_HUB_TOKEN` or from the standard shared cache
  file `~/.cache/huggingface/token`. That file may have been created by another
  Hugging Face tool. DiffusionBear does not create it and does not use tokens
  from other services.

You can remove the tokens in **Parameters → Secrets** or by deleting the token
files while the app is closed.

## What is stored on your Mac

The standalone app uses `~/Library/Application Support/DiffusionBear/` by
default. The gallery/model store can be relocated; the app writes the chosen
absolute path into `data_path` and `asset_path` next to that folder.

| Location | Contents |
| --- | --- |
| `~/Library/Application Support/DiffusionBear/data/` | Default app data: `settings.json`, `loras.json`, generated images and JSON sidecars, uploaded reference images/masks, prompt cache, queue recovery files, engine pid files, token files and downloaded models/LoRAs. |
| `~/Library/Application Support/DiffusionBear/data/generated/` | Generated images and their JSON sidecars. |
| `~/Library/Application Support/DiffusionBear/data/uploads/` | Reference images and masks you add. |
| `~/Library/Application Support/DiffusionBear/data/prompt_cache.sqlite` | Local text-encoder cache. It contains prompt text and/or prompt hashes and embeddings for faster reuse; it never leaves the Mac. |
| `~/Library/Application Support/DiffusionBear/data/models/` | Downloaded model weights. |
| `~/Library/Application Support/DiffusionBear/data/lora_files/` and `data/SDXL/` | LoRA files and SDXL assets. |
| `~/Library/Application Support/DiffusionBear/Logs/backend.log` | Backend log for the current run. It is truncated at launch. |
| `~/Library/Application Support/DiffusionBear/Logs/launch.log` | Launcher diagnostics. |
| `~/.cache/huggingface/` | Hugging Face downloads and cache. Shared with other Hugging Face tools on the same Mac. |
| `~/Library/WebKit/` and the app's WebKit data | Local WebKit cache and the theme preference kept in `localStorage`. The app window only loads the local backend; external links open in your browser. |

If you relocate the data or model store, the files above live in the folders
you chose instead.

The app does not create a user account or a cloud copy. Deleting the app data
folder removes the gallery, settings, tokens, cache and downloaded models that
live there. Deleting `~/.cache/huggingface/` also affects other Hugging Face
apps on the same Mac.

## Details inside your images

By default, generated images carry prompt and generation settings in PNG `tEXt`
chunks and EXIF, following the Civitai/Automatic1111 convention. That can
include the prompt, negative prompt, model and LoRA names, sampler, size,
steps, guidance, seed, generation time and the artist credit (default:
“DiffusionBear”). File paths are not included.

Anyone you share an image with can read that metadata. Enable **Stealth mode**
(no metadata) in **Parameters → Defaults** or per generation to save clean
files. Sidecar JSON files in the gallery still contain the generation data
locally.

## Local server and security

The backend listens on loopback only (`127.0.0.1`/`localhost`) and rejects
non-loopback clients. It uses an optional local API token for non-GET requests
when configured. It is not exposed to your local network.

The app opens external links in your normal browser. It does not embed remote
web pages.

## Third-party services

Hugging Face and Civitai process requests under their own privacy policies.
DiffusionBear does not control what they log or retain. Their privacy policies
are available at:

- Hugging Face: <https://huggingface.co/privacy>
- Civitai: <https://civitai.com>

Model and LoRA authors may also have their own terms. Check the model card and
the in-app licence summary before downloading or using a model.

## Changes

If the app's network behaviour changes, this file will be updated in the same
release. The current version describes the code in this repository and the
standalone app built from it.

## Contact

Privacy questions can be opened as a normal issue at
<https://github.com/OuincheWinch/DiffusionBear/issues>. For a sensitive report,
use the private vulnerability flow described in [SECURITY.md](SECURITY.md).
