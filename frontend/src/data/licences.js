export const GITHUB_REPO_URL = "https://github.com/OuincheWinch/MLX-Diffusion";
export const GITHUB_LICENSE_URL = `${GITHUB_REPO_URL}/blob/main/LICENSE`;
export const AUTHOR_WEBSITE = "https://www.ouinche.com";
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
    packages: [
      {
        name: "FLUX.2-klein 4B (mlx-community quant)",
        license: "Black Forest Labs — Non-Commercial ⚠",
        url: "https://huggingface.co/mlx-community/FLUX.2-Klein-4B-mlx",
        caution: true,
      },
      {
        name: "Krea 2 Turbo",
        license: "Apache-2.0",
        url: "https://github.com/krea-ai/krea-2",
      },
      {
        name: "Z-Image-Turbo",
        license: "Apache-2.0",
        url: "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo",
      },
      {
        name: "Qwen-Image 2.1 (experimental)",
        license: "Apache-2.0",
        url: "https://huggingface.co/Qwen/Qwen-Image-2.1",
      },
      {
        name: "Juggernaut XL Lightning",
        license: "RunDiffusion — Non-Commercial ⚠",
        url: "https://www.civitai.com/models/133005",
        caution: true,
      },
      {
        name: "Qwen2.5-0.5B-Instruct (prompt enhancer)",
        license: "Apache-2.0",
        url: "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct",
      },
    ],
  },
];