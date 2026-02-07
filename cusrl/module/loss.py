import math
from typing import Literal

import torch
from torch import Tensor, nn

__all__ = ["GradientPenaltyLoss", "L2RegularizationLoss", "NormalNllLoss"]


class GradientPenaltyLoss(nn.Module):
    r"""Computes the gradient penalty loss.

    This loss calculates the squared L2 norm of the gradients of the outputs
    with respect to the inputs. It is commonly utilized in WGAN-GP and AMP
    (Adversarial Motion Priors) to enforce 1-Lipschitz continuity on the
    discriminator by constraining the gradient norm.

    .. math::
        \text{loss} = \mathbb{E}[\|\nabla_{\text{inputs}} \text{outputs}\|^2]

    Args:
        reduction ({"none", "mean", "sum"}, optional):
            Specifies the reduction to apply to the output. Defaults to
            ``"mean"``.

    Note:
        The ``inputs`` tensor must have ``requires_grad=True`` set before
        computing ``outputs`` to enable gradient calculation.
    """

    def __init__(self, reduction: Literal["none", "mean", "sum"] = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, outputs: Tensor, inputs: Tensor) -> Tensor:
        """
        Args:
            outputs (Tensor): The output of the computation graph.
            inputs (Tensor): The input to the computation graph.
        """
        gradients = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        penalty = gradients.square().sum(dim=-1)

        if self.reduction == "mean":
            return penalty.mean()
        if self.reduction == "sum":
            return penalty.sum()
        return penalty


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
            ``False``.
        eps (float, optional):
            Value used to clamp variance for stability. Defaults to ``1e-6``.
        reduction ({"none", "mean", "sum"}, optional):
            Specifies the reduction to apply to the output. Defaults to
            ``"mean"``.
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


class L2RegularizationLoss(nn.Module):
    r"""Computes the L2 regularization loss.

    This loss calculates the squared L2 norm of the input tensor.

    .. math::
        \text{loss} = || \text{input} ||_2^2

    Args:
        reduction ({"none", "mean", "sum"}, optional):
            Specifies the reduction to apply to the output. Defaults to
            ``"mean"``.
    """

    def __init__(self, reduction: Literal["none", "mean", "sum"] = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, input: Tensor) -> Tensor:
        loss = input.square()

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
