import torch
from torch import Tensor, nn
from torch.nn.functional import linear

__all__ = ["LinearFp32", "disable_autocast"]


def disable_autocast(device_type: str):
    return torch.autocast(device_type=device_type, enabled=False)


class LinearFp32(nn.Linear):
    def forward(self, input: Tensor) -> Tensor:
        with disable_autocast(input.device.type):
            bias = self.bias.float() if self.bias is not None else None
            return linear(input.float(), self.weight.float(), bias)
