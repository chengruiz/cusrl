from dataclasses import dataclass
from typing import TypedDict

from einops import rearrange
from torch import Tensor, nn

from cusrl.nn.module.module import Module, ModuleFactory
from cusrl.nn.utils.recurrent import (
    compute_sequence_lengths,
    gather_memory,
    scatter_memory,
    select_initial_memory,
    split_and_pad_sequences,
    unpad_and_merge_sequences,
)
from cusrl.utils.nest import map_nested
from cusrl.utils.typing import Memory

__all__ = ["Gru", "Lstm", "Rnn", "VanillaRnn"]


class _VanillaRnn(nn.RNN):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            bias=bias,
            dropout=dropout,
        )

    def forward(
        self,
        input: Tensor,
        memory: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if memory is None:
            output, hn = super().forward(input)
        else:
            # input: [ L, N, C ]
            # memory: [ N, K * C ]
            h0 = rearrange(memory, "n (k c) -> k n c", k=self.num_layers, c=self.hidden_size)
            output, hn = super().forward(input, h0.contiguous())
        hn = rearrange(hn, "k n c -> n (k c)")
        return output, hn


class LstmMemory(TypedDict):
    hidden: Tensor
    cell: Tensor


class _Lstm(nn.LSTM):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            dropout=dropout,
        )

    def forward(
        self,
        input: Tensor,
        memory: LstmMemory | None = None,
    ) -> tuple[Tensor, LstmMemory]:
        if memory is None:
            output, (hn, cn) = super().forward(input)
        else:
            # input: [ L, N, C ]
            # memory: [ N, K * C ]
            h0, c0 = memory["hidden"], memory["cell"]
            assert h0.shape == c0.shape, "Hidden and cell states must have the same shape"
            h0 = rearrange(h0, "n (k c) -> k n c", k=self.num_layers, c=self.hidden_size)
            c0 = rearrange(c0, "n (k c) -> k n c", k=self.num_layers, c=self.hidden_size)
            output, (hn, cn) = super().forward(input, (h0.contiguous(), c0.contiguous()))

        hn = rearrange(hn, "k n c -> n (k c)")
        cn = rearrange(cn, "k n c -> n (k c)")
        return output, {"hidden": hn, "cell": cn}


class _Gru(nn.GRU):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            dropout=dropout,
        )

    def forward(
        self,
        input: Tensor,
        memory: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if memory is None:
            output, hn = super().forward(input)
        else:
            # input: [ L, N, C ]
            # memory: [ N, K * C ] or [ L, N, K * C ]
            h0 = rearrange(memory, "n (k c) -> k n c", k=self.num_layers, c=self.hidden_size)
            output, hn = super().forward(input, h0.contiguous())
        hn = rearrange(hn, "k n c -> n (k c)")
        return output, hn


@dataclass
class RnnFactory(ModuleFactory["Rnn"]):
    module_type: str
    hidden_size: int
    num_layers: int = 1
    nonlinearity: str = "tanh"
    bias: bool = True
    dropout: float = 0.0

    def __call__(self, input_dim: int | None = None, output_dim: int | None = None):
        assert input_dim is not None
        module_type = self.module_type.lower()
        if module_type == "rnn" or module_type == "vanilla":
            module = _VanillaRnn(
                input_size=input_dim,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                nonlinearity=self.nonlinearity,
                bias=self.bias,
                dropout=self.dropout,
            )
        elif module_type == "lstm":
            module = _Lstm(
                input_size=input_dim,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                bias=self.bias,
                dropout=self.dropout,
            )
        elif module_type == "gru":
            module = _Gru(
                input_size=input_dim,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                bias=self.bias,
                dropout=self.dropout,
            )
        else:
            raise ValueError(f"Unsupported RNN module class '{self.module_type}'")
        return Rnn(module, output_dim=output_dim)


class Rnn(Module):
    """Wrapper around the recurrent modules defined in this file.

    This module provides a unified interface for the internal RNN adapters
    (``_VanillaRnn``, ``_Lstm``, and ``_Gru``), handling single tensors,
    sequences with termination signals, and packed sequences.

    It also manages recurrent memory, including resetting states for finished
    episodes within a batch when ``done`` flags are provided.

    Args:
        rnn (_VanillaRnn | _Lstm | _Gru):
            An instantiated recurrent module created by this module's helpers
            or factories.
        output_dim (int | None, optional):
            Output feature dimension. If not ``None``, a linear projection is
            applied to the recurrent output. Defaults to ``None``.
    """

    Factory = RnnFactory

    def __init__(self, rnn: _VanillaRnn | _Lstm | _Gru, output_dim: int | None = None):
        super().__init__(rnn.input_size, output_dim or rnn.hidden_size, is_recurrent=True)
        self.rnn = rnn
        self.output_proj = nn.Linear(rnn.hidden_size, output_dim) if output_dim else nn.Identity()

    def forward(
        self,
        input: Tensor,
        memory: Memory = None,
        *,
        done: Tensor | None = None,
        sequential: bool = True,
        pack_sequence: bool = False,
        **kwargs,
    ) -> tuple[Tensor, Memory]:
        """Forward pass through the recurrent neural network.

        This method handles both single-steped or sequential data. It resets the
        recurrent state for finished episodes within a sequence when the `done`
        tensor is provided.

        Args:
            input (Tensor):
                Input tensor of shape :math:`(L, N, ...)` if ``sequential`` else
                :math:`(N, ...)`, where :math:`L` is the sequence length,
                :math:`N` is the batch size.
            memory (Memory, optional):
                The recurrent state from the previous step. Defaults to None,
                which initializes a zero state.
            done (Tensor | None, optional):
                A boolean tensor of shape :math:`(L, N, 1)` indicating
                terminations. If provided, the memory is reset for the
                corresponding batch entries where `done` is True. Requires
                ``sequential`` to be True. Defaults to None.
            sequential (bool):
                If True, the input is treated as a sequences. Otherwise, it's
                treated as a single batch of data. Defaults to True.
            pack_sequence (bool):
                If True and ``done`` is provided, the input sequence is packed
                to preserve the final recurrent state. Defaults to False.

        Outputs:
            - **output** (Tensor):
                The output tensor of shape :math:`(L, N, ...)` if ``sequential``
                else :math:`(N, ...)`.
            - **memory** (Memory):
                The updated recurrent state.
        """
        if sequential and input.dim() >= 3:
            memory = select_initial_memory(memory, input.shape[:-1])
        if done is not None:
            if not sequential:
                raise ValueError("'done' can be provided only when 'sequential' is True")
            latent, memory = self._forward_sequence(input, memory, done, pack_sequence=pack_sequence)
        else:
            latent, memory = self._forward_tensor(input, memory, sequential=sequential)
        return self.output_proj(latent), memory

    def _forward_tensor(
        self,
        input: Tensor,
        memory: Memory = None,
        sequential: bool = True,
    ) -> tuple[Tensor, Memory]:
        reshaped_input, reshaped_memory = self._reshape_input(input, memory, sequential=sequential)
        reshaped_latent, reshaped_output_memory = self.rnn(reshaped_input, reshaped_memory)
        return self._reshape_output(reshaped_latent, reshaped_output_memory, input.shape, sequential=sequential)

    def _forward_sequence(
        self,
        input: Tensor,
        memory: Memory,
        done: Tensor,
        pack_sequence: bool = False,
    ) -> tuple[Tensor, Memory]:
        padded_input, mask = split_and_pad_sequences(input, done)
        scattered_memory = scatter_memory(memory, done)
        if pack_sequence:
            if input.dim() != 3:
                raise ValueError(f"Packed RNN input must be 3D, but got {input.dim()} dimensions")
            sequence_lens = compute_sequence_lengths(done)
            reshaped_padded_input, reshaped_scattered_memory = self._reshape_input(padded_input, scattered_memory)
            reshaped_packed_input = nn.utils.rnn.pack_padded_sequence(
                reshaped_padded_input, lengths=sequence_lens.cpu(), enforce_sorted=False
            )
            reshaped_packed_latent, reshaped_scattered_output_memory = self.rnn(
                reshaped_packed_input, reshaped_scattered_memory
            )
            reshaped_padded_latent, _ = nn.utils.rnn.pad_packed_sequence(
                reshaped_packed_latent,
                total_length=reshaped_padded_input.size(0),
            )
            padded_latent, scattered_output_memory = self._reshape_output(
                reshaped_padded_latent, reshaped_scattered_output_memory, padded_input.shape
            )
            output_memory = gather_memory(scattered_output_memory, done)
        else:
            padded_latent, _ = self._forward_tensor(padded_input, scattered_memory)
            # Without packed sequences, the RNN also consumes padded timesteps,
            # so the returned final state no longer matches the last valid
            # timestep of each episode and cannot be used as output memory.
            output_memory = None
        latent = unpad_and_merge_sequences(padded_latent, mask)
        return latent, output_memory

    def _reshape_input(
        self,
        input: Tensor,
        memory: Memory,
        sequential: bool = True,
    ) -> tuple[Tensor, Memory]:
        if input.dim() < 3:
            # ( C )    -> ( 1, 1, C )
            # ( N, C ) -> ( 1, N, C )
            input = input.reshape(1, -1, input.size(-1))
            if memory is not None:
                memory = map_nested(lambda mem: mem.reshape(input.size(1), -1), memory)
        if input.dim() >= 3:
            # ( L, N, C )      -> ( L, N, C )       if ``sequential`` else ( 1, L * N, C )
            # ( L, N, ..., C ) -> ( L, N * ..., C ) if ``sequential`` else ( 1, L * N * ..., C )
            input = input.reshape(input.size(0) if sequential else 1, -1, input.size(-1))
            if memory is not None:
                memory = map_nested(lambda mem: mem.flatten(0, -2), memory)
        return input, memory

    def _reshape_output(
        self,
        output: Tensor,
        memory: Memory,
        original_input_shape: tuple[int, ...],
        sequential: bool = True,
    ) -> tuple[Tensor, Memory]:
        output = output.reshape(*original_input_shape[:-1], output.size(-1))
        if memory is not None:
            batch_dims = original_input_shape[(1 if sequential and len(original_input_shape) > 2 else 0) : -1]
            memory = map_nested(lambda mem: mem.reshape(*batch_dims, mem.size(-1)), memory)
        return output, memory

    def step_memory(self, input: Tensor, memory: Memory = None, sequential: bool = True, **kwargs):
        original_input_shape = input.shape
        if sequential and input.dim() >= 3:
            memory = select_initial_memory(memory, input.shape[:-1])
        input, memory = self._reshape_input(input, memory, sequential=sequential)
        latent, memory = self.rnn(input, memory)
        _, memory = self._reshape_output(latent, memory, original_input_shape, sequential=sequential)
        return memory


@dataclass
class VanillaRnnFactory(ModuleFactory["VanillaRnn"]):
    hidden_size: int
    num_layers: int = 1
    nonlinearity: str = "tanh"
    bias: bool = True
    dropout: float = 0.0

    def __call__(self, input_dim: int, output_dim: int | None = None):
        return VanillaRnn(input_dim=input_dim, output_dim=output_dim, **self.__dict__)


class VanillaRnn(Rnn):
    Factory = VanillaRnnFactory

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        bias: bool = True,
        dropout: float = 0.0,
        output_dim: int | None = None,
    ):
        super().__init__(
            _VanillaRnn(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                nonlinearity=nonlinearity,
                bias=bias,
                dropout=dropout,
            ),
            output_dim=output_dim,
        )


@dataclass
class LstmFactory(ModuleFactory["Lstm"]):
    hidden_size: int
    num_layers: int = 1
    bias: bool = True
    dropout: float = 0.0

    def __call__(self, input_dim: int, output_dim: int | None = None):
        return Lstm(input_dim=input_dim, output_dim=output_dim, **self.__dict__)


class Lstm(Rnn):
    Factory = LstmFactory

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        output_dim: int | None = None,
    ):
        super().__init__(
            _Lstm(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=bias,
                dropout=dropout,
            ),
            output_dim=output_dim,
        )


@dataclass
class GruFactory(ModuleFactory["Gru"]):
    hidden_size: int
    num_layers: int = 1
    bias: bool = True
    dropout: float = 0.0

    def __call__(self, input_dim: int, output_dim: int | None = None):
        return Gru(input_dim=input_dim, output_dim=output_dim, **self.__dict__)


class Gru(Rnn):
    Factory = GruFactory

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        output_dim: int | None = None,
    ):
        super().__init__(
            _Gru(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=bias,
                dropout=dropout,
            ),
            output_dim=output_dim,
        )
