# Third-Party Notices

DiffusionBear source code is MIT-licensed. This file records the third-party
code and model families used by the release.

## mflux / vanch007 MOSS-Image MLX architecture

- Original project: https://github.com/filipstrand/mflux
- Copyright (c) 2026 Filip Strand
- License: MIT
- Vendored/adapted branch: https://github.com/vanch007/mflux
  - Branch: `feature/moss-image-sft`
  - Commit: `b1fe66cc62109f374fe7d713411c589d3810d0c0`
- Local files: `backend/moss_mlx/`
- Adaptations: relative imports, Q4 checkpoint mapping, selective MLX
  quantization, small FLUX.2 VAE decoder, local MLX Qwen text encoder, and the
  three-stage memory lifecycle.
- Full MIT text: `backend/moss_mlx/LICENSE`; file-level attribution:
  `backend/moss_mlx/NOTICE`.

## MOSS-Image model weights

- Upstream: https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT
- Revision: `f972bd4b4b9e3852133cc5da954d87bc32cd8fb9`
- License: Apache License 2.0
- Q4 MLX conversion: https://huggingface.co/mflux-community/moss-image-sft-mflux-q4
- DiffusionBear does not redistribute model weights. They are downloaded on
  demand and remain subject to their upstream licences.

## Qwen 3.5 text encoder

- Base model: Qwen 3.5, Alibaba Cloud / Qwen Team
- MOSS text-encoder conversion: https://huggingface.co/Ouinche/moss-image-qwen3.5-te-mlx-q8g64
- Local directory: `qwen3_5-te-q8g64`
- The Qwen model licence applies. DiffusionBear ships no Qwen weights.

## MOSS DMD adapters

- 4-step: `darask0/MOSS-Image_4-step_adapter_lora`
- 8-step: `lodestones/MOSS-Image-SFT-8-steps-dmd`
- Adapter licences remain those of their publishers. They are downloaded on
  demand, not vendored.

## MLX, mlx-lm, transformers, tokenizers, Hugging Face Hub

These are runtime dependencies and are not vendored in this repository. Their
respective upstream licences apply.

## Other model families

FLUX.1/2/3, SDXL derivatives, Qwen-Image, Krea, Z-Image, FIBO, and their LoRAs
are downloaded from Hugging Face or Civitai and retain their own licences. The
app's **Licences** tab displays the applicable terms.

## ControlNet branches (optional, downloaded once on first ControlNet use)

- FLUX.1-schnell ControlNet-Canny (q4 branch):
  https://huggingface.co/mflux-community/flux-1-schnell-controlnet-canny-mflux-q4
  - Revision: `e23a38491871d04ba193abb90fc76fd29ee9ea10`
  - License: Apache License 2.0 (base: black-forest-labs/FLUX.1-schnell, Apache-2.0).
    The InstantX FLUX.1-dev ControlNet pack is deliberately NOT used: it is
    flux-1-dev-non-commercial-licensed.
- Qwen-Image 2.1 Turbo ControlNet-Union (q4 branch):
  https://huggingface.co/mflux-community/qwen-image-2-1-controlnet-union-mflux-q4
  - Revision: `75b32be6f1a5caa826f696c795a1610c0a1755be`
  - License: Qwen Research License (base: Qwen/Qwen-Image-2.1).
- Z-Image-Turbo ControlNet-Union 2.1 (q4 branch):
  https://huggingface.co/mflux-community/z-image-turbo-controlnet-union-2-1-mflux-q4
  - Revision: `f7a4ab9f5f96bf48ce5c3b9fc2ff626656bcd91b`
  - License: Apache License 2.0 (base: Tongyi-MAI/Z-Image-Turbo; upstream
    alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 is Apache-2.0).
- DiffusionBear does not redistribute these weights. They are fetched from
  Hugging Face on first ControlNet use (with "Allow required downloads" ticked)
  and remain subject to their upstream licences.
