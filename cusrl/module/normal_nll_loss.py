import math
from typing import Literal

from torch import Tensor, nn

__all__ = ["NormalNllLoss"]


LOG_SQRT_2PI = math.log(2 * math.pi) / 2


class NormalNllLoss(nn.Module):
    r"""Computes the negative log-likelihood (NLL) of a Normal distribution
    parameterized by mean and log variance.

    The input tensor is expected to have its last dimension split evenly into
    mean ($\mu$) and log variance ($\log \sigma^2$) parts.

    For each element the loss is computed as:
    .. math::
        \text{loss} = \frac{1}{2} \left(
            \log \sigma^2 + \frac{(\text{target} - \mu)^2}{\sigma^2}
        \right) + \frac{1}{2} \log(2\pi)

    Args:
        full (bool, optional):
            Includes the constant term in the loss computation. Defaults to
            `False`.
        eps (float, optional):
            Value used to clamp variance for stability. Defaults to 1e-6.
        reduction ({"none", "mean", "sum"}, optional):
            Specifies the reduction to apply to the output.
    """

    def __init__(
        self,
        *,
        full: bool = False,
        eps: float = 1e-6,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ) -> None:
        if eps <= 0:
            raise ValueError("'eps' must be greater than zero.")

        super().__init__()
        self.full = full
        self.log_eps = math.log(eps)
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        mean, log_var = input.chunk(2, dim=-1)
        log_var = log_var.clamp_min(self.log_eps)
        nll = 0.5 * (log_var + (target - mean).square() / log_var.exp())
        if self.full:
            nll = nll + LOG_SQRT_2PI

        if self.reduction == "mean":
            return nll.mean()
        if self.reduction == "sum":
            return nll.sum()
        return nll
