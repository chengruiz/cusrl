from torch import Tensor, nn
from torch.nn.functional import gelu, silu

__all__ = ["GeGlu", "SwiGlu"]


class GeGlu(nn.GLU):
    def forward(self, input: Tensor) -> Tensor:
        x, gate = input.chunk(2, dim=self.dim)
        return x * gelu(gate)


class SwiGlu(nn.GLU):
    def forward(self, input: Tensor) -> Tensor:
        x, gate = input.chunk(2, dim=self.dim)
        return x * silu(gate)
