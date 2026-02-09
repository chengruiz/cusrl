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
    parameterized by mean and a variance parameter.

    The input tensor is expected to have its last dimension split evenly into
    mean ($\mu$) and the variance parameter specified by ``mode``.

    For each element the loss is computed as:
    .. math::
        \text{loss} = \frac{1}{2} \left(
            \log \sigma^2 + \frac{(\text{target} - \mu)^2}{\sigma^2}
        \right) + \frac{1}{2} \log(2\pi)

    Args:
        mode ({"log_var", "log_std", "var", "std"}, optional):
            Specifies the representation of the variance parameter. Defaults to
            ``"log_var"``.
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
        mode: Literal["log_var", "log_std", "var", "std"] = "log_var",
        full: bool = False,
        eps: float = 1e-6,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ) -> None:
        if eps <= 0:
            raise ValueError("'eps' must be greater than zero.")
        if mode not in {"log_var", "log_std", "var", "std"}:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of 'log_var', 'log_std', 'var', or 'std'.")

        super().__init__()
        self.mode = mode
        self.full = full
        self.eps = eps
        self.sqrt_eps = math.sqrt(eps)
        self.log_eps = math.log(eps)
        self.reduction = reduction

    def forward(self, input: Tensor | tuple[Tensor, Tensor], target: Tensor) -> Tensor:
        if isinstance(input, tuple):
            mean, dist = input
        else:
            mean, dist = input.chunk(2, dim=-1)

        if self.mode == "log_var":
            log_var = dist.clamp_min(self.log_eps)
            var = log_var.exp()
        elif self.mode == "log_std":
            log_std = dist.clamp_min(self.log_eps / 2)
            log_var = log_std * 2
            var = log_var.exp()
        elif self.mode == "var":
            var = dist.clamp_min(self.eps)
            log_var = var.log()
        elif self.mode == "std":
            std = dist.clamp_min(self.sqrt_eps)
            var = std.square()
            log_var = std.log() * 2
        else:
            raise ValueError(f"Invalid mode '{self.mode}'.")

        nll = 0.5 * (log_var + (target - mean).square() / var)
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
