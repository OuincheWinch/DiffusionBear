"""Bear default custom sampler: fixed-mu exponential-shift Euler for mflux.

Subclasses the built-in FlowMatchEulerDiscreteScheduler and pins the
exponential shift exponent mu instead of the resolution-derived empirical
value. Higher mu keeps more noise late in the schedule (crisper high-res
detail, softer subjects); lower mu does the opposite. This file is the
protected reference: delete it, and the app re-extracts it at startup.
Copy it as a template for your own samplers.
"""
SAMPLER = {
    "name": "bear_euler_shift",
    "label": "Bear fixed-mu Euler (mflux)",
    "engines": ["mflux"],
    "mflux": "BearEulerShiftScheduler",
}

try:
    import mlx.core as mx  # noqa: F401
    from mflux.models.common.schedulers.flow_match_euler_discrete_scheduler import (
        FlowMatchEulerDiscreteScheduler,
    )
except Exception:  # pragma: no cover - validated on the mflux runtime
    FlowMatchEulerDiscreteScheduler = object


class BearEulerShiftScheduler(FlowMatchEulerDiscreteScheduler):
    """Flow-match Euler with a pinned exponential shift exponent."""

    MU = 1.1

    def set_image_seq_len(self, image_seq_len: int) -> None:
        sigmas = mx.linspace(
            1.0,
            1.0 / self.config.num_inference_steps,
            self.config.num_inference_steps,
            dtype=mx.float32,
        )
        sigmas = self._time_shift_exponential_array(self.MU, 1.0, sigmas)
        self._timesteps = sigmas * self.num_train_timesteps
        sigmas = mx.concatenate([sigmas, mx.zeros((1,), dtype=sigmas.dtype)], axis=0)
        self._sigmas = sigmas
