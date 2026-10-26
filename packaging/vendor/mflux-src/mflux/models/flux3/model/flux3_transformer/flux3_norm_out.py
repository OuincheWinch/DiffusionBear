import mlx.core as mx
from mlx import nn

from mflux.models.common.config import ModelConfig


class Flux3AdaLayerNorm(nn.Module):
    """Adaptive LayerNorm for Black Forest Labs FLUX.3 LastLayer.

    Crucial ordering distinction vs Diffusers AdaLayerNormContinuous:
    - Diffusers AdaLayerNormContinuous: chunk 0 is scale, chunk 1 is shift.
    - Black Forest Labs LastLayer: chunk 0 is shift, chunk 1 is scale
      (see BFL dit.py: `shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)`
       and `(1.0 + scale) * F.layer_norm(x, ...) + shift`).
    Inverting these chunks injects the raw multiplicative scale vector as a constant
    additive bias across all sequence tokens, which projects into proj_out as a massive
    per-quadrant checkerboard artifact (16-px period).
    """

    def __init__(self, embedding_dim: int, conditioning_embedding_dim: int, bias: bool = False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        self.norm = nn.LayerNorm(dims=embedding_dim, eps=1e-6, affine=False)

    def __call__(self, x: mx.array, text_embeddings: mx.array) -> mx.array:
        text_embeddings = self.linear(nn.silu(text_embeddings).astype(ModelConfig.precision))
        chunk_size = self.embedding_dim
        shift = text_embeddings[:, 0 * chunk_size : 1 * chunk_size]
        scale = text_embeddings[:, 1 * chunk_size : 2 * chunk_size]
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x
