import json
import math
import struct

import mlx.core as mx


class IrisFewStepScheduler:
    # The sampler of few-step LoRAs distilled from the base model (DMD2, as in pentacoxian-dev/iris-3b-turbo-lora):
    # steps evenly spaced from pure noise, shifted, one forward per step without CFG, and fresh noise between steps.
    # Such a LoRA records turbo_steps and shift in its safetensors metadata, which is how the reference sampler
    # (pentaCoxian/iris-3b-turbo, generate_turbo) knows to switch to it.
    NUM_TRAIN_TIMESTEPS = 1000
    MAX_STEPS = 100

    def __init__(self, num_inference_steps: int, shift: float, seed: int):
        grid = [1.0 - i / num_inference_steps for i in range(num_inference_steps)]
        self.sigmas = [shift * s / (1 + (shift - 1) * s) for s in grid]
        self.num_inference_steps = num_inference_steps
        self.key = mx.random.key(seed)

    @staticmethod
    def from_lora_files(paths: list[str] | None, scales: list[float] | None = None) -> tuple[int, float] | None:
        # (steps, shift) of the first LoRA that is a few-step adapter, read from the file header only. A LoRA at scale
        # 0 changes no weight, so it does not change the sampler either.
        scales = scales or [1.0] * len(paths or [])
        for path, scale in zip(paths or [], scales):
            extra = IrisFewStepScheduler._metadata(path).get("extra", {})
            if "turbo_steps" in extra and scale != 0:
                return IrisFewStepScheduler._settings(path, extra)
        return None

    @staticmethod
    def _settings(path: str, extra: dict) -> tuple[int, float]:
        try:
            steps, shift = int(extra["turbo_steps"]), float(extra.get("shift", 4.0))
        except (TypeError, ValueError):
            steps, shift = 0, 0.0
        if not 1 <= steps <= IrisFewStepScheduler.MAX_STEPS or not math.isfinite(shift) or shift <= 0:
            raise ValueError(f"{path} records turbo_steps={extra.get('turbo_steps')!r} and shift={extra.get('shift')!r}; a few-step LoRA needs 1 to {IrisFewStepScheduler.MAX_STEPS} steps and a positive shift.")  # fmt: off
        return steps, shift

    def model_time(self, step: int) -> mx.array:
        return mx.array([self.sigmas[step]], dtype=mx.float32) * IrisFewStepScheduler.NUM_TRAIN_TIMESTEPS

    def step(self, latents: mx.array, velocity: mx.array, step: int) -> tuple[mx.array, mx.array]:
        # Returns the next state and the clean prediction (the final image on the last step)
        clean = latents - self.sigmas[step] * velocity
        if step + 1 == self.num_inference_steps:
            return clean, clean
        self.key, noise_key = mx.random.split(self.key)
        noise = mx.random.normal(latents.shape, dtype=mx.float32, key=noise_key)
        next_sigma = self.sigmas[step + 1]
        return (1 - next_sigma) * clean + next_sigma * noise, clean

    @staticmethod
    def refuse_turbo_svd(paths: list[str] | None) -> None:
        # Aikimi/iris-3b-turbo-svd-lora: an SVD of turbo - base plus dense.* tensors that replace base ones, sampled
        # with the Turbo times. Loaded as a LoRA it drops the dense tensors and gives a broken image
        for path in paths or []:
            header = IrisFewStepScheduler._header(path)
            svd = (header.get("__metadata__") or {}).get("format") == "iris-turbo-svd-v1"
            if svd or any(key.startswith("dense.") for key in header):
                raise ValueError(f"{path} is an Iris Turbo SVD file (iris-turbo-svd-v1, dense.* tensors), which mflux cannot load yet. Use --model iris-3b for Turbo.")  # fmt: off

    @staticmethod
    def _header(path: str) -> dict:
        with open(path, "rb") as file:
            length = struct.unpack("<Q", file.read(8))[0]
            return json.loads(file.read(length))

    @staticmethod
    def _metadata(path: str) -> dict:
        metadata = dict(IrisFewStepScheduler._header(path).get("__metadata__") or {})
        # extra is free-form JSON: anything but an object ("null", a list) records no sampler
        try:
            extra = json.loads(metadata.get("extra", "{}"))
        except (TypeError, ValueError):
            extra = {}
        metadata["extra"] = extra if isinstance(extra, dict) else {}
        return metadata
