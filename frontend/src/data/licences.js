export const GITHUB_REPO_URL = "https://github.com/OuincheWinch/DiffusionBear";
export const GITHUB_LICENSE_URL = `${GITHUB_REPO_URL}/blob/main/LICENSE`;
export const AUTHOR_WEBSITE = "https://www.ouinche.com";
export const APP_WEBSITE = "https://www.DiffusionBear.com";
export const AI_CREDITS = ["Gemini", "OpenCode", "Big Pickle"];

export const LICENCE_SECTIONS = [
  {
    id: "backend",
    titleKey: "licences.sectionBackend",
    noteKey: "licences.noteBackend",
    packages: [
      { name: "FastAPI", license: "MIT", url: "https://github.com/fastapi/fastapi" },
      { name: "Uvicorn", license: "BSD-3-Clause", url: "https://github.com/encode/uvicorn" },
      { name: "Pydantic", license: "MIT", url: "https://github.com/pydantic/pydantic" },
      { name: "python-multipart", license: "Apache-2.0", url: "https://github.com/Kludex/python-multipart" },
      { name: "MLX", license: "MIT (Apple)", url: "https://github.com/ml-explore/mlx" },
      { name: "mflux", license: "MIT", url: "https://github.com/filipstrand/mflux", extra: "locally patched fork (Qwen-Image 2.1 support, PR #736 + 16GB weight-mapping/TE fixes)" },
      { name: "mlx-lm", license: "MIT", url: "https://github.com/ml-explore/mlx-lm" },
      { name: "mlx-taef", license: "MIT", url: "https://github.com/IonDen/mlx-taef" },
      { name: "mlx-teacache", license: "Apache-2.0", url: "https://github.com/IonDen/mlx-teacache" },
      { name: "Pillow", license: "MIT-CMU", url: "https://github.com/python-pillow/Pillow" },
      { name: "pillow-heif", license: "BSD-3-Clause", url: "https://github.com/bigcat88/pillow_heif" },
      { name: "safetensors", license: "Apache-2.0", url: "https://github.com/huggingface/safetensors" },
      { name: "NumPy", license: "BSD-3-Clause", url: "https://github.com/numpy/numpy" },
      { name: "huggingface-hub", license: "Apache-2.0", url: "https://github.com/huggingface/huggingface_hub" },
    ],
  },
  {
    id: "sdxl",
    titleKey: "licences.sectionSdxl",
    noteKey: "licences.noteSdxl",
    packages: [
      { name: "torch (conversion only)", license: "BSD-3-Clause", url: "https://github.com/pytorch/pytorch" },
      { name: "diffusers (conversion only)", license: "Apache-2.0", url: "https://github.com/huggingface/diffusers" },
      { name: "transformers (conversion only)", license: "Apache-2.0", url: "https://github.com/huggingface/transformers" },
      { name: "accelerate (conversion only)", license: "Apache-2.0", url: "https://github.com/huggingface/accelerate" },
    ],
  },
  {
    id: "frontend",
    titleKey: "licences.sectionFrontend",
    noteKey: "licences.noteFrontend",
    packages: [
      { name: "React + react-dom", license: "MIT", url: "https://github.com/facebook/react" },
      { name: "Vite", license: "MIT", url: "https://github.com/vitejs/vite" },
      { name: "@vitejs/plugin-react", license: "MIT", url: "https://github.com/vitejs/vite-plugin-react" },
      { name: "oxlint (dev)", license: "MIT", url: "https://github.com/oxc-project/oxc" },
    ],
  },
  {
    id: "models",
    titleKey: "licences.sectionModels",
    noteKey: "licences.noteWeights",
    // Licence names below are what the upstream model card or licence file
    // states, verified 2026-11-03. A quantised conversion repo is not the
    // authority: its metadata has been wrong (FLUX 9B, Krea 2, Qwen 2.1,
    // Z-Image). Always check the upstream model. A page stating "other" (or
    // nothing) gets NO licence value -- the table renders the unknown-licence
    // fallback instead. Never restore a paraphrase like "Vendor — Non-Commercial":
    // test_licence_center.py fails the build on any name outside the verified set.
    //
    // outputs = can the generated image be used commercially.
    // weights = can the model be used commercially (including paid client work
    //           and paid hosting). These are different questions.
    packages: [
      {
        name: "FLUX.2-klein 4B",
        license: "Apache-2.0",
        url: "https://huggingface.co/black-forest-labs/FLUX.2-klein-4B",
        outputs: "yes",
        weights: "yes",
        conditionKey: "licences.conditionApache",
      },
      {
        name: "FLUX.2-klein 9B",
        license: "FLUX Non-Commercial License",
        url: "https://huggingface.co/black-forest-labs/FLUX.2-klein-9B",
        outputs: "yes",
        weights: "no",
        conditionKey: "licences.conditionFluxNine",
        caution: true,
      },
      {
        name: "FLUX.3-klein 5.6B",
        license: "FLUX Kommunity License v1.0",
        url: "https://huggingface.co/black-forest-labs/flux-3-image-klein",
        outputs: "conditional",
        weights: "conditional",
        conditionKey: "licences.conditionFluxKommunity",
        caution: true,
      },
      {
        name: "FLUX.1-schnell",
        license: "Apache-2.0",
        url: "https://huggingface.co/black-forest-labs/FLUX.1-schnell",
        outputs: "yes",
        weights: "yes",
        conditionKey: "licences.conditionApache",
      },
      {
        name: "MOSS-Image Base 5B",
        license: "Apache-2.0",
        url: "https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base",
        outputs: "yes",
        weights: "yes",
        conditionKey: "licences.conditionApache",
      },
      {
        name: "MOSS-Image SFT 5B",
        license: "Apache-2.0",
        url: "https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT",
        outputs: "yes",
        weights: "yes",
        conditionKey: "licences.conditionApache",
      },
      {
        name: "Z-Image Turbo 6B",
        license: "Apache-2.0",
        url: "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo",
        outputs: "yes",
        weights: "yes",
        conditionKey: "licences.conditionApache",
        extra: "upstream licence; the mflux conversion repo mislabels it as Tongyi Qianwen",
      },
      {
        name: "Qwen-Image Edit",
        license: "Apache-2.0",
        url: "https://huggingface.co/Qwen/Qwen-Image-Edit",
        outputs: "yes",
        weights: "yes",
        conditionKey: "licences.conditionApache",
      },
      {
        name: "Qwen-Image 2.1 (MLX q4, experimental)",
        license: "Qwen Research License",
        url: "https://huggingface.co/Qwen/Qwen-Image-2.1",
        outputs: "no",
        weights: "no",
        conditionKey: "licences.conditionQwenResearch",
        caution: true,
      },
      {
        name: "Qwen-Image 2.1 Turbo (MLX q4, experimental)",
        license: "Qwen Research License",
        url: "https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo",
        outputs: "no",
        weights: "no",
        conditionKey: "licences.conditionQwenResearch",
        caution: true,
      },
      {
        name: "Krea 2 Turbo 13B",
        license: "Krea 2 Community License",
        url: "https://www.krea.ai/krea-2-licensing",
        outputs: "conditional",
        weights: "conditional",
        conditionKey: "licences.conditionKrea",
        caution: true,
      },
      {
        name: "FIBO 2.0 Edit Turbo",
        license: "CC BY-NC 4.0",
        url: "https://huggingface.co/briaai/Fibo-2.0-edit-turbo",
        outputs: "no",
        weights: "no",
        conditionKey: "licences.conditionFibo",
        caution: true,
      },
      {
        name: "Juggernaut XL Lightning (MLX)",
        license: "CreativeML OpenRAIL-M",
        url: "https://huggingface.co/RunDiffusion/Juggernaut-XL-Lightning",
        outputs: "yes",
        weights: "yes",
        conditionKey: "licences.conditionOpenRail",
      },
      {
        name: "RealVisXL V5.0 Lightning (MLX)",
        license: "OpenRAIL++",
        url: "https://huggingface.co/SG161222/RealVisXL_V5.0_Lightning",
        outputs: "yes",
        weights: "yes",
        conditionKey: "licences.conditionOpenRail",
      },
      {
        name: "RealVisXL V5.0 Standard / Hyper-SD",
        license: "OpenRAIL++",
        url: "https://huggingface.co/SG161222/RealVisXL_V5.0",
        outputs: "yes",
        weights: "yes",
        conditionKey: "licences.conditionOpenRail",
      },
      {
        name: "Juggernaut XI v11 (MLX)",
        license: "CC BY-NC-ND 4.0",
        url: "https://huggingface.co/RunDiffusion/Juggernaut-XI-v11",
        outputs: "no",
        weights: "no",
        conditionKey: "licences.conditionJuggernautXI",
        caution: true,
      },
      {
        name: "Pony Diffusion V6 XL",
        license: "Modified Fair AI Public License 1.0-SD",
        url: "https://civitai.com/models/257749/pony-diffusion-v6-xl",
        outputs: "yes",
        weights: "conditional",
        conditionKey: "licences.conditionPony",
        caution: true,
      },
      {
        name: "Qwen2.5-1.5B-Instruct (prompt enhancer)",
        license: "Apache-2.0",
        url: "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct",
        outputs: "yes",
        weights: "yes",
        conditionKey: "licences.conditionApache",
      },
      {
        name: "Qwen2.5-0.5B-Instruct (prompt enhancer)",
        license: "Apache-2.0",
        url: "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct",
        outputs: "yes",
        weights: "yes",
        conditionKey: "licences.conditionApache",
      },
    ],
  },
];

const BUILTIN_MODEL_IDS = {
  "fibo2-turbo": "FIBO 2.0 Edit Turbo",
  "flux1-schnell": "FLUX.1-schnell",
  "flux2-klein-4b": "FLUX.2-klein 4B",
  "flux2-klein-9b": "FLUX.2-klein 9B",
  "flux3-klein": "FLUX.3-klein 5.6B",
  "juggernaut-xi": "Juggernaut XI v11 (MLX)",
  "juggernaut-xl-lightning": "Juggernaut XL Lightning (MLX)",
  "krea2-turbo": "Krea 2 Turbo 13B",
  "moss-image-base": "MOSS-Image Base 5B",
  "moss-image-sft": "MOSS-Image SFT 5B",
  "pony": "Pony Diffusion V6 XL",
  "qwen-image-2.1": "Qwen-Image 2.1 (MLX q4, experimental)",
  "qwen-image-2.1-turbo": "Qwen-Image 2.1 Turbo (MLX q4, experimental)",
  "qwen-image-edit": "Qwen-Image Edit",
  "realvis-xl-v5": "RealVisXL V5.0 Standard / Hyper-SD",
  "realvis-xl-v5-lightning": "RealVisXL V5.0 Lightning (MLX)",
  "z-image-turbo": "Z-Image Turbo 6B",
};

const BUILTIN_MODELS = LICENCE_SECTIONS.find((section) => section.id === "models")?.packages || [];

export function sortByName(items, nameOf = (item) => item?.name) {
  const read = (item) => String(nameOf(item) ?? "").trim();
  return [...items].sort((a, b) =>
    read(a).localeCompare(read(b), undefined, {
      numeric: true,
      sensitivity: "base",
    }),
  );
}

export function knownLicenceFor(model) {
  const id = String(model?.id || "").trim().toLowerCase();
  const name = BUILTIN_MODEL_IDS[id];
  if (!name) return null;
  return BUILTIN_MODELS.find((pkg) => pkg.name === name) || null;
}

export function licenceSummaryFor(model) {
  const known = knownLicenceFor(model);
  const license = model?.license || null;
  const rawIdentifier = license?.identifier;
  const identifier =
    typeof rawIdentifier === "string" && rawIdentifier && rawIdentifier !== "other"
      ? rawIdentifier
      : null;
  const rawUrl = known?.url || license?.source_url || license?.url;
  const url =
    typeof rawUrl === "string" && /^https?:\/\//i.test(rawUrl.trim())
      ? rawUrl.trim()
      : null;
  return {
    known,
    license,
    name: known?.license || identifier,
    url,
    outputs: known?.outputs || null,
    weights: known?.weights || null,
    conditionKey: known?.conditionKey || null,
    caution: Boolean(known?.caution),
  };
}