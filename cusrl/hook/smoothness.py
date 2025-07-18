from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from cusrl.template import Hook
from cusrl.utils.recurrent import split_and_pad_sequences

__all__ = ["ActionSmoothnessLoss"]


class ActionSmoothnessLoss(Hook):
    """A hook to penalize non-smooth actions in temporal sequences.

    This hook calculates a loss based on the 1st and/or 2nd order differences
    of the action sequence, effectively penalizing high action velocities and
    accelerations.

    The loss is computed using 1D convolution with fixed kernels:
    - 1st order (velocity): `[-1, 1]`
    - 2nd order (acceleration): `[-1, 2, -1]`

    Args:
        weight_1st_order (torch.Tensor | None, optional):
            Weight for the 1st order smoothness loss. Can be a scalar or a tensor
            matching the action dimension. Defaults to None.
        weight_2nd_order (torch.Tensor | None, optional):
            Weight for the 2nd order smoothness loss. Can be a scalar or a tensor
            matching the action dimension. Defaults to None.
    """

    weight_1st_order: torch.Tensor | None
    weight_2nd_order: torch.Tensor | None
    conv_1st_order: torch.Tensor
    conv_2nd_order: torch.Tensor
    MUTABLE_ATTRS = ["weight_1st_order", "weight_2nd_order"]

    def __init__(
        self,
        weight_1st_order: float | Sequence[float] | None = None,
        weight_2nd_order: float | Sequence[float] | None = None,
    ):
        self.weight_1st_order = None if weight_1st_order is None else torch.tensor(weight_1st_order)
        self.weight_2nd_order = None if weight_2nd_order is None else torch.tensor(weight_2nd_order)

    def init(self):
        if self.weight_1st_order is not None:
            self.weight_1st_order = self.weight_1st_order.to(self.agent.device)
        if self.weight_2nd_order is not None:
            self.weight_2nd_order = self.weight_2nd_order.to(self.agent.device)
        self.conv_1st_order = self.agent.to_tensor([[[-1.0, 1.0]]])
        self.conv_2nd_order = self.agent.to_tensor([[[-1.0, 2.0, -1.0]]])

    def objective(self, batch: dict[str, Any]):
        if batch["curr_action_mean"].ndim != 3:
            raise ValueError("Expected batch to be temporal.")
        if batch["curr_action_mean"].size(0) < 3:
            seq_len = batch["curr_action_mean"].size(0)
            raise ValueError(f"Expected sequences to have at least 3 time steps, but got {seq_len}.")

        padded_action, mask = split_and_pad_sequences(batch["curr_action_mean"], batch["done"])
        action_sequence = padded_action.permute(1, 2, 0).flatten(0, 1).unsqueeze(1)  # [N * C, 1, T]
        smoothness_loss = None
        if self.weight_1st_order is not None:
            smoothness_1st_order = (
                # convolve at time dimension
                nn.functional.conv1d(action_sequence, self.conv_1st_order)  # [N * C, 1, T-1]
                .reshape(*padded_action.shape[1:], -1)  # [N, C, T-1]
                .permute(2, 0, 1)  # [T-1, N, C]
            )
            smoothness_1st_order_loss = (self.weight_1st_order * smoothness_1st_order[mask[1:]].abs()).mean()
            smoothness_loss = smoothness_1st_order_loss
            self.agent.record(smoothness_1st_order_loss=smoothness_1st_order_loss)

        if self.weight_2nd_order is not None:
            smoothness_2nd_order = (
                nn.functional.conv1d(action_sequence, self.conv_2nd_order)  # [ N * C, 1, T-2 ]
                .reshape(*padded_action.shape[1:], -1)  # [N, C, T-2]
                .permute(2, 0, 1)  # [T-2, N, C]
            )
            smoothness_loss_2nd_order = (self.weight_2nd_order * smoothness_2nd_order[mask[2:]].abs()).mean()
            smoothness_loss = (
                smoothness_loss_2nd_order if smoothness_loss is None else smoothness_loss + smoothness_loss_2nd_order
            )
            self.agent.record(smoothness_2nd_order_loss=smoothness_loss_2nd_order)

        return smoothness_loss

    def update_attribute(self, name, value):
        if name == "weight_1st_order":
            self.weight_1st_order = self.agent.to_tensor(value)
        elif name == "weight_2nd_order":
            self.weight_2nd_order = self.agent.to_tensor(value)
        else:
            raise ValueError(f"Attribute '{name}' is not mutable for hook {self.name}.")
