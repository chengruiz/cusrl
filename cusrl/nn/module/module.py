from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Generic, Optional, TypeAlias

import torch
from torch import nn
from typing_extensions import TypeVar

from cusrl.utils.misc import import_obj
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


ModuleT = TypeVar("ModuleT", bound="Module", default="Module")
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
    for handling recurrent states, managing device placement, and supporting
    distributed training.

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
    """

    Factory = ModuleFactory

    def __init__(
        self,
        input_dim: int | None = None,
        output_dim: int | None = None,
        is_recurrent: bool = False,
        like: Optional["Module"] = None,
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

    def forward_with_memory(
        self,
        input: torch.Tensor,
        memory: Memory = None,
        *,
        done: torch.Tensor | None = None,
        sequential: bool = True,
        **kwargs,
    ) -> tuple[Any, Memory]:
        """Executes the module and always returns an ``(output, memory)`` tuple.

        For recurrent modules, ``memory``, ``done``, and ``sequential`` are
        forwarded to :meth:`__call__`. For non-recurrent modules, :meth:`__call__`
        is invoked without recurrent arguments and the input ``memory`` is
        returned unchanged alongside the output.
        """
        if self.is_recurrent:
            return self(input, memory=memory, done=done, sequential=sequential, **kwargs)
        return self(input, **kwargs), memory

    def step_memory(self, input: torch.Tensor, memory: Memory = None, **kwargs) -> Memory:
        """Updates the module's memory based on the input.

        This method is useful in scenarios where only the next memory state is
        needed, without the full output of the forward pass. The default
        implementation calls :meth:`forward_with_memory`. Subclasses can
        override this for a more efficient update if computing the memory state
        is less expensive than a full forward pass.
        """
        if not self.is_recurrent:
            return None
        _, next_memory = self.forward_with_memory(input, memory=memory, **kwargs)
        return next_memory

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

    @contextmanager
    def collect_repr(self, *names: str) -> Iterator[dict[str, Any]]:
        reprs: dict[str, Any] = {}
        handles = [
            self.get_submodule(name).register_forward_hook(
                lambda _module, _args, output, key=name: reprs.__setitem__(
                    key, output[0] if isinstance(output, tuple) else output
                )
            )
            for name in dict.fromkeys(names)
        ]
        try:
            yield reprs
        finally:
            for handle in handles:
                handle.remove()

    def inference(self, memory: Memory = None):
        from cusrl.nn.module.inference import InferenceWrapper

        return InferenceWrapper(self, memory=memory)


def resolve_activation_fn(activation_fn: str | type[nn.Module]) -> type[nn.Module]:
    if isinstance(activation_fn, str):
        activation_name = activation_fn
        if "." not in activation_name:
            activation_name = f"torch.nn.{activation_name}"
        module_name, qual_name = activation_name.split(".", 1)
        try:
            activation_fn = import_obj(module_name, qual_name)
        except ImportError:
            raise ValueError(f"No activation function named '{activation_name}' was found")
    if not issubclass(activation_fn, nn.Module):
        raise TypeError(f"Activation functions must be subclasses of nn.Module; got {activation_fn}")
    return activation_fn
