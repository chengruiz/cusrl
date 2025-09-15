from torch import Tensor, nn

__all__ = ["DetachGradient"]


class DetachGradient(nn.Module):
    """A module that detaches the gradient during the backward pass."""

    def forward(self, input: Tensor) -> Tensor:
        return input.detach()
