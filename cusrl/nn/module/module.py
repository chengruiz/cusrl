from collections.abc import Callable
from typing import Any, Generic, Optional, TypeAlias, TypeVar

import torch
from torch import nn

from cusrl.utils.nest import iterate_nested
from cusrl.utils.typing import Memory, Slice

__all__ = [
    "Module",
    "ModuleFactory",
    "ModuleFactoryLike",
    "ModuleT",
    "LayerFactoryLike",
    "LayerT",
    "resolve_activation_fn",
]


ModuleT = TypeVar("ModuleT", bound="Module")
LayerT = TypeVar("LayerT", bound=nn.Module)


class ModuleFactory(Generic[ModuleT]):
    def __call__(self, input_dim: int | None = None, output_dim: int | None = None) -> ModuleT:
        raise NotImplementedError


ModuleFactoryLike: TypeAlias = Callable[[int | None, int | None], "Module"] | ModuleFactory
LayerFactoryLike: TypeAlias = Callable[[int | None, int | None], nn.Module] | ModuleFactoryLike


class Module(nn.Module):
    """A base class for all cusrl modules, extending :class:`torch.nn.Module`.

    This class provides a standardized interface and additional functionalities
    for all neural network modules within the framework. It includes features
    for handling recurrent states, managing device placement, supporting
    distributed training, and intermediate representation storage.

    Args:
        input_dim (int | None, optional):
            The dimensionality of the input. Required if ``like`` is not
            provided. Defaults to ``None``.
        output_dim (int | None, optional):
            The dimensionality of the output. Required if ``like`` is not
            provided. Defaults to ``None``.
        is_recurrent (bool, optional):
            Whether the module is recurrent. Defaults to ``False``.
        like (Optional[Module], optional):
            Another module instance from which to copy ``input_dim``,
            ``output_dim``, and ``is_recurrent`` attributes. Defaults to
            ``None``.
        intermediate_repr (dict[str, Any] | None, optional):
            An initial dictionary for intermediate representations. Defaults to
            ``None``.
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
                raise ValueError("'input_dim' and 'output_dim' must be specified when 'like' is not provided")
            if input_dim <= 0:
                raise ValueError("'input_dim' must be a positive integer")
            if output_dim <= 0:
                raise ValueError("'output_dim' must be a positive integer")
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.is_recurrent: bool = is_recurrent
        self.intermediate_repr: dict[str, Any] = intermediate_repr or {}
        self._rnn_compatible: bool = False

    @property
    def device(self) -> torch.device:
        param = next(self.parameters(), None)
        if param is not None:
            return param.device
        buffer = next(self.buffers(), None)
        if buffer is not None:
            return buffer.device
        return torch.device("cpu")

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
                The memory to be reset. Can be a single tensor or a nested
                structure of tensors. If None, the method does nothing.
            done (SliceType | torch.Tensor | None, optional):
                A mask or slice indicating which parts of the memory to reset.
                If it's a boolean tensor, it should correspond to the
                environments that are finished. Defaults to ``None``.
        """
        if memory is None:
            return
        if isinstance(done, torch.Tensor):
            done = done.squeeze(-1)
        elif done is None:
            done = slice(None)

        for _, tensor in iterate_nested(memory):
            tensor[done] = 0

    def clear_intermediate_repr(self):
        self.intermediate_repr.clear()

    def inference(self, memory: Memory = None):
        from cusrl.nn.module.inference import InferenceWrapper

        return InferenceWrapper(self, memory=memory)

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


def resolve_activation_fn(activation_fn: str | type[nn.Module]) -> type[nn.Module]:
    if isinstance(activation_fn, str):
        activation_name = activation_fn.removeprefix("torch.").removeprefix("nn.")
        activation_fn = getattr(nn, activation_name, None)
        if activation_fn is None:
            raise ValueError(f"No activation function named '{activation_name}' was found in torch.nn")
    if not issubclass(activation_fn, nn.Module):
        raise TypeError(f"Activation functions must be subclasses of nn.Module; got {activation_fn}")
    return activation_fn
