from typing import Any

import torch
from torch import nn

__all__ = ["ParameterWrapper"]


class ParameterWrapper(nn.Module):
    """A minimal torch.nn.Module that wraps a tensor as a learnable parameter.

    This module registers the provided tensor as a torch.nn.Parameter so it
    participates in gradient-based optimization. During the forward pass, it
    simply returns this parameter unchanged.

    Args:
        data (torch.Tensor):
            Initial value for the parameter. Its shape, dtype, and device are
            preserved when wrapped as a torch.nn.Parameter.
    """

    def __init__(self, data: torch.Tensor | Any):
        super().__init__()
        self.param = nn.Parameter(torch.as_tensor(data))

    def forward(self):
        return self.param
