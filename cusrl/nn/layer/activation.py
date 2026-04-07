from torch import Tensor, nn
from torch.nn.functional import gelu, silu

__all__ = ["GeGlu", "SwiGlu"]


class GeGlu(nn.GLU):
    r"""
    .. math::
        \text{GeGLU}(a, b) = a \odot \text{GELU}(b)
    """

    def forward(self, input: Tensor) -> Tensor:
        x, gate = input.chunk(2, dim=self.dim)
        return x * gelu(gate)


class SwiGlu(nn.GLU):
    r"""
    .. math::
        \text{SwiGLU}(a, b) = a \odot \text{SiLU}(b)
    """

    def forward(self, input: Tensor) -> Tensor:
        x, gate = input.chunk(2, dim=self.dim)
        return x * silu(gate)
