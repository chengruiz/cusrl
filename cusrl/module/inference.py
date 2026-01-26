from typing import Any

import numpy as np
import torch

from cusrl.module.module import Module
from cusrl.utils.typing import ArrayType, Memory, Slice

__all__ = ["InferenceWrapper"]


class InferenceWrapper(Module):
    """A wrapper module designed to facilitate inference with CusRL modules.

    This class wraps a given CusRL module, handling memory management for
    recurrent states and ensuring seamless conversion between NumPy arrays and
    PyTorch tensors during the forward pass.

    Args:
        module (Module):
            The CusRL module to be wrapped.
        memory (Memory, optional):
            The initial hidden state for the Module. Defaults to ``None``.
        forward_kwargs (dict[str, Any] | None, optional):
            Additional keyword arguments to be passed to the module's forward
            method. Defaults to ``None``.
    """

    def __init__(
        self,
        module: Module,
        memory: Memory = None,
        forward_kwargs: dict[str, Any] | None = None,
    ):
        module = module.rnn_compatible()
        super().__init__(like=module, intermediate_repr=module.intermediate_repr)
        self._wrapped = module
        self.memory = memory
        self.forward_kwargs = forward_kwargs or {}

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
        action, self.memory = self._wrapped(
            input,
            memory=self.memory,
            **self.forward_kwargs,
            **kwargs,
        )
        if add_batch_dim:
            action = action.squeeze(0)
        return action

    def reset(self, indices: Slice = slice(None)):
        self._wrapped.reset_memory(self.memory, indices)
