import math
from typing import TYPE_CHECKING

import mlx.core as mx

from mflux.models.common.schedulers.base_scheduler import BaseScheduler

if TYPE_CHECKING:
    from mflux.models.common.config.config import Config


class IrisTrainingSchedule(BaseScheduler):
    # The noise levels mflux-train draws from for Iris: an even grid of NUM_TRAIN_TIMESTEPS levels from pure noise (the
    # adapter's config asks for that many steps), shifted as the reference trainer shifts its draws, by 2 at 256x256, 3 at 512 and 4 at 1024 (the stage configs of speridlabs/iris-3b). The default
    # uniform draw over this grid is what mflux recommends: the reference trainer's logit-normal draw ("timestep_type":
    # "sigmoid") rarely reaches the highest noise levels, and in style LoRAs it learned the colors less reliably.
    NUM_TRAIN_TIMESTEPS = 1000

    def __init__(self, config: "Config"):
        steps = config.num_inference_steps
        shift = IrisTrainingSchedule.shift(config.height, config.width)
        raw = [1.0 - i / steps for i in range(steps)]
        self._sigmas = mx.array([shift * s / (1 + (shift - 1) * s) for s in raw] + [0.0], dtype=mx.float32)

    @property
    def sigmas(self) -> mx.array:
        return self._sigmas

    @staticmethod
    def shift(height: int, width: int) -> float:
        # The reference covers 256 to 1024. Below 128x128 the formula drops under 1 and reaches 0 at 64x64, which
        # divides 0 by 0 at the top of the grid, so smaller images get no shift
        return max(1.0, 2.0 + math.log2(math.sqrt(height * width) / 256))

    def step(self, noise: mx.array, timestep: int, latents: mx.array, **kwargs) -> mx.array:  # noqa: ARG002
        # Euler on the flow, for completeness; training only reads the sigmas
        return latents + (self._sigmas[timestep + 1] - self._sigmas[timestep]) * noise
