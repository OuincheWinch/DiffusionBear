export const GITHUB_REPO_URL = "https://github.com/OuincheWinch/DiffusionBear";
export const GITHUB_LICENSE_URL = `${GITHUB_REPO_URL}/blob/main/LICENSE`;
export const AUTHOR_WEBSITE = "https://www.ouinche.com";
export const APP_WEBSITE = "https://www.DiffusionBear.com";
export const AI_CREDITS = ["Gemini", "0xAlpha", "Big Pickle"];

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
      { name: "mlx_diffuser", license: "CC0-1.0", url: "https://github.com/AmirHossein-razlighi/mlx_diffuser" },
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
    // Licence names below are what each linked page states, verified 2026-10-15.
    // A page stating "other" (or nothing) gets NO licence value -- the table
    // renders the unknown-licence fallback instead. Never restore a paraphrase
    // like "Vendor — Non-Commercial": test_licence_center.py fails the build
    // on any name outside the verified set.
    packages: [
      {
        name: "FLUX.2-klein 4B",
        license: "apache-2.0",
        url: "https://huggingface.co/mlx-community/flux2-klein-4b-4bit",
      },
      {
        name: "FLUX.2-klein 9B",
        license: "apache-2.0",
        url: "https://huggingface.co/mlx-community/flux2-klein-9b-4bit",
      },
      {
        name: "Juggernaut XL Lightning (MLX)",
        license: "creativeml-openrail-m",
        url: "https://huggingface.co/RunDiffusion/Juggernaut-XL-Lightning",
      },
      {
        name: "RealVisXL V5.0 Lightning (MLX)",
        license: "openrail++",
        url: "https://huggingface.co/SG161222/RealVisXL_V5.0_Lightning",
      },
      {
        name: "RealVisXL V5.0 Standard / Hyper-SD",
        license: "openrail++",
        url: "https://huggingface.co/SG161222/RealVisXL_V5.0",
      },
      {
        name: "Juggernaut XI v11 (MLX)",
        license: "cc-by-nc-nd-4.0",
        url: "https://huggingface.co/RunDiffusion/Juggernaut-XI-v11",
        caution: true,
      },
      {
        name: "Z-Image Turbo 6B",
        license: "apache-2.0",
        url: "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo",
      },
      {
        name: "Krea 2 Turbo 13B",
        license: "Apache-2.0",
        url: "https://github.com/krea-ai/krea-2",
      },
      {
        name: "Qwen-Image 2.1 (MLX q4, experimental)",
        url: "https://huggingface.co/mlx-community/Qwen-Image-2.1-MLX-4bit",
      },
      {
        name: "Qwen-Image 2.1 Turbo (MLX q4, experimental)",
        url: "https://huggingface.co/Sawfwair/Qwen-Image-2.1-Turbo-MLX-Mixed-4bit",
      },
      {
        name: "Qwen2.5-1.5B-Instruct (prompt enhancer)",
        license: "Apache-2.0",
        url: "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct",
      },
    ],
  },
];