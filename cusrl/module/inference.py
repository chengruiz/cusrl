from collections.abc import Callable

import numpy as np
import torch

from cusrl.module.module import Module
from cusrl.utils.typing import Array, ArrayType, Memory, Slice

__all__ = ["InferenceModule"]


class InferenceModule(Module):
    def __init__(self, module: Module, memory: Memory = None):
        module = module.rnn_compatible()
        super().__init__(like=module, intermediate_repr=module.intermediate_repr)
        self._wrapped = module
        self.memory = memory

        self._forward_call: Callable[..., Array]
        object.__setattr__(self, "_forward_call", module.forward)  # avoid registering as submodule
        self._forward_kwargs = {}

    @property
    def wrapped(self):
        return self._wrapped

    @staticmethod
    def _decorator_forward__preserve_io_format(act_method):
        def wrapped_forward(self, input: ArrayType, **kwargs) -> ArrayType:
            is_numpy = isinstance(input, np.ndarray)
            input_tensor = torch.as_tensor(input)
            device, dtype = input_tensor.device, input_tensor.dtype
            input_tensor = input_tensor.to(self.device)
            output = act_method(self, input_tensor, **kwargs)
            output = output.to(device=device, dtype=dtype)
            return output.numpy() if is_numpy else output

        return wrapped_forward

    @_decorator_forward__preserve_io_format
    @torch.no_grad()
    def forward(self, input: torch.Tensor, **kwargs):
        add_batch_dim = input.ndim == 1
        if add_batch_dim:
            input = input.unsqueeze(0)
        action, self.memory = self._forward_call(
            input,
            memory=self.memory,
            **self._forward_kwargs,
            **kwargs,
        )
        if add_batch_dim:
            action = action.squeeze(0)
        return action

    def reset(self, indices: Slice = slice(None)):
        self._wrapped.reset_memory(self.memory, indices)
