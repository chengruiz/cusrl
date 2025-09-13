from collections.abc import Sequence
from typing import cast

import torch
from torch import nn

from cusrl.module.distribution import MeanStdDict
from cusrl.template import Hook
from cusrl.utils.recurrent import split_and_pad_sequences

__all__ = ["ActionSmoothnessLoss"]


class ActionSmoothnessLoss(Hook):
    """Penalizes non-smooth actions in temporal sequences.

    This hook calculates a loss based on the 1st and/or 2nd order differences
    of the action sequence, effectively penalizing high action velocities and
    accelerations.

    The loss is computed using 1D convolution with fixed kernels:
    - 1st order (velocity): `[-1, 1]`
    - 2nd order (acceleration): `[-1, 2, -1]`

    Args:
        weight_1st_order (float | Sequence[float] | None, optional):
            Weight for the 1st order smoothness loss. Could be a scalar or a
            tensor matching the action dimension. Defaults to None.
        weight_2nd_order (float | Sequence[float] | None, optional):
            Weight for the 2nd order smoothness loss. Could be a scalar or a
            tensor matching the action dimension. Defaults to None.
    """

    def __init__(
        self,
        weight_1st_order: float | Sequence[float] | None = None,
        weight_2nd_order: float | Sequence[float] | None = None,
    ):
        super().__init__()

        # Mutable attributes
        self.weight_1st_order: float | Sequence[float] | None
        self.weight_2nd_order: float | Sequence[float] | None
        self.register_mutable("weight_1st_order", weight_1st_order)
        self.register_mutable("weight_2nd_order", weight_2nd_order)

        # Runtime attributes
        self._weight1_tensor: torch.Tensor | None
        self._weight2_tensor: torch.Tensor | None
        self.conv_1st_order: torch.Tensor
        self.conv_2nd_order: torch.Tensor

    def init(self):
        self._weight1_tensor = None if self.weight_1st_order is None else self.agent.to_tensor(self.weight_1st_order)
        self._weight2_tensor = None if self.weight_2nd_order is None else self.agent.to_tensor(self.weight_2nd_order)
        self.conv_1st_order = self.agent.to_tensor([[[-1.0, 1.0]]])
        self.conv_2nd_order = self.agent.to_tensor([[[-1.0, 2.0, -1.0]]])

    def objective(self, batch):
        action_mean = cast(MeanStdDict, batch["curr_action_dist"])["mean"]
        if action_mean.ndim != 3:
            raise ValueError("Expected batch to be temporal.")
        if (seq_len := action_mean.size(0)) < 3:
            raise ValueError(f"Expected sequences to have at least 3 time steps, but got {seq_len}.")

        padded_action, mask = split_and_pad_sequences(action_mean, batch["done"])
        action_sequence = padded_action.permute(1, 2, 0).flatten(0, 1).unsqueeze(1)  # [N * C, 1, T]
        smoothness_loss = None
        if self._weight1_tensor is not None:
            smoothness_1st_order = (
                # Convolve at time dimension
                nn.functional.conv1d(action_sequence, self.conv_1st_order)  # [N * C, 1, T-1]
                .reshape(*padded_action.shape[1:], -1)  # [N, C, T-1]
                .permute(2, 0, 1)  # [T-1, N, C]
            )
            smoothness_1st_order_loss = (self._weight1_tensor * smoothness_1st_order[mask[1:]].abs()).mean()
            smoothness_loss = smoothness_1st_order_loss
            self.agent.record(smoothness_1st_order_loss=smoothness_1st_order_loss)

        if self._weight2_tensor is not None:
            smoothness_2nd_order = (
                nn.functional.conv1d(action_sequence, self.conv_2nd_order)  # [ N * C, 1, T-2 ]
                .reshape(*padded_action.shape[1:], -1)  # [N, C, T-2]
                .permute(2, 0, 1)  # [T-2, N, C]
            )
            smoothness_loss_2nd_order = (self._weight2_tensor * smoothness_2nd_order[mask[2:]].abs()).mean()
            smoothness_loss = (
                smoothness_loss_2nd_order if smoothness_loss is None else smoothness_loss + smoothness_loss_2nd_order
            )
            self.agent.record(smoothness_2nd_order_loss=smoothness_loss_2nd_order)

        return smoothness_loss

    def update_attribute(self, name, value):
        super().update_attribute(name, value)
        if name == "weight_1st_order":
            self._weight1_tensor = None if value is None else self.agent.to_tensor(value)
        elif name == "weight_2nd_order":
            self._weight2_tensor = None if value is None else self.agent.to_tensor(value)
