from collections.abc import Callable
from typing import Any, Generic, Optional, TypeAlias, TypeVar

import torch
from torch import nn
from typing_extensions import Self

from cusrl.utils.typing import Memory, Slice

__all__ = [
    "DistributedDataParallel",
    "Module",
    "ModuleType",
    "ModuleFactory",
    "ModuleFactoryLike",
    "LayerFactoryLike",
]


class DistributedDataParallel(nn.parallel.DistributedDataParallel):
    is_distributed = True

    def __init__(self, module, *args, **kwargs):
        kwargs["gradient_as_bucket_view"] = True
        super().__init__(module, *args, **kwargs)
        self.register_state_dict_post_hook(self._state_dict_post_hook)
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            return getattr(self.module, item)

    @staticmethod
    def _state_dict_post_hook(module: nn.Module, state_dict: dict[str, Any], prefix: str, *args):
        for key in tuple(state_dict.keys()):
            if key.startswith(prefix):
                new_key = f"{prefix}{key.removeprefix(f'{prefix}module.')}"
                state_dict[new_key] = state_dict.pop(key)

    def _load_state_dict_pre_hook(self, state_dict: dict[str, Any], prefix: str, *args):
        for key in tuple(state_dict.keys()):
            if key.startswith(prefix):
                new_key = f"{prefix}module.{key.removeprefix(prefix)}"
                state_dict[new_key] = state_dict.pop(key)


ModuleType = TypeVar("ModuleType", bound="Module")


class ModuleFactory(Generic[ModuleType]):
    def __call__(self, input_dim: int | None, output_dim: int | None) -> ModuleType:
        raise NotImplementedError

    @staticmethod
    def _resolve_activation_fn(activation_fn: str | type[nn.Module]) -> type[nn.Module]:
        if isinstance(activation_fn, str):
            activation_fn = getattr(nn, activation_fn, None)
            if activation_fn is None:
                raise ValueError(f"Activation function '{activation_fn}' not found in nn module.")
        if not issubclass(activation_fn, nn.Module):
            raise TypeError(f"Activation function must be a subclass of nn.Module, got {activation_fn}.")
        return activation_fn


ModuleFactoryLike: TypeAlias = Callable[[int | None, int | None], "Module"] | ModuleFactory
LayerFactoryLike: TypeAlias = Callable[[int, int], nn.Module] | ModuleFactoryLike


class Module(nn.Module):
    """A base class for all `cusrl` modules, extending `torch.nn.Module`.

    This class provides a standardized interface and additional functionalities
    for all neural network modules within the framework. It includes features
    for handling recurrent states, managing device placement, supporting
    distributed training, and intermediate representation storage.

    Args:
        input_dim (int | None, optional):
            The dimensionality of the input. Required if `like` is not provided.
            Defaults to None.
        output_dim (int | None, optional):
            The dimensionality of the output. Required if `like` is not
            provided. Defaults to None.
        is_recurrent (bool, optional):
            Whether the module is recurrent. Defaults to False.
        like (Optional["Module"], optional):
            Another module instance from which to copy `input_dim`,
            `output_dim`, and `is_recurrent` attributes. Defaults to None.
        intermediate_repr (dict[str, Any] | None, optional):
            An initial dictionary for intermediate representations. Defaults to
            None.
    """

    Factory = ModuleFactory

    def __init__(
        self,
        input_dim: int | None = None,
        output_dim: int | None = None,
        is_recurrent: bool = False,
        like: Optional["Module"] = None,
        intermediate_repr: dict[str, Any] | None = None,
    ):
        super().__init__()
        if like is not None:
            input_dim = like.input_dim
            output_dim = like.output_dim
            is_recurrent = like.is_recurrent
        else:
            if input_dim is None or output_dim is None:
                raise ValueError("'input_dim' and 'output_dim' should be specified if 'like' is not provided.")
            if input_dim <= 0:
                raise ValueError("'input_dim' should be positive integers.")
            if output_dim <= 0:
                raise ValueError("'output_dim' should be a positive integer.")
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.is_recurrent: bool = is_recurrent
        self.is_distributed: bool = False
        self.intermediate_repr: dict[str, Any] = intermediate_repr or {}
        self._rnn_compatible: bool = False

    @property
    def device(self):
        if hasattr(self, "_device"):
            return self._device
        return next(self.parameters()).device

    def to_distributed(self) -> DistributedDataParallel | Self:
        if not self.is_distributed:
            return DistributedDataParallel(self)
        return self

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def step_memory(self, input: torch.Tensor, memory: Memory = None, **kwargs) -> Memory:
        """Updates the module's memory based on the input.

        This method is useful in scenarios where only the next memory state is
        needed, without the full output of the forward pass. The default
        implementation calls the `forward` method. Subclasses can override this
        for a more efficient update if computing the memory state is less
        expensive than a full forward pass.
        """

        if not self.is_recurrent:
            return None
        _, *next_memory = self(input, memory=memory, **kwargs)
        return next_memory[0]

    def reset_memory(self, memory: Memory, done: Slice | torch.Tensor | None = None):
        """Resets the memory of the model for finished environments.

        This method is typically called at the end of an environment step to
        reset the recurrent memory (e.g., hidden states of an RNN) for those
        environments that have finished.

        Args:
            memory (MemoryType):
                The memory to be reset. Can be a single tensor or a tuple of
                tensors. If None, the method does nothing.
            done (SliceType | torch.Tensor | None, optional):
                A mask or slice indicating which parts of the memory to reset.
                If it's a boolean tensor, it should correspond to the
                environments that are finished. Defaults to None.
        """
        if memory is None:
            return
        if isinstance(memory, tuple):
            for mem in memory:
                self.reset_memory(mem, done)
            return
        if isinstance(done, torch.Tensor):
            done = done.squeeze(-1)
        memory[..., done, :] = 0

    def clear_intermediate_repr(self):
        self.intermediate_repr.clear()

    def inference(self, memory=None):
        from cusrl.module.inference import InferenceModule

        return InferenceModule(self, memory=memory)

    def rnn_compatible(self):
        if not self.is_recurrent and not self._rnn_compatible:
            self._rnn_compatible = True
            original_forward = self.forward

            def wrapped_forward(input, **kwargs):
                output = original_forward(input)
                if "memory" in kwargs:
                    return output, kwargs["memory"]
                return output

            self.forward = wrapped_forward
        return self
