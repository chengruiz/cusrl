from typing import TypeAlias

import torch
from torch import nn

from cusrl.module.module import Module, ModuleFactory
from cusrl.utils.recurrent import compute_sequence_lengths, split_and_pad_sequences, unpad_and_merge_sequences
from cusrl.utils.typing import Memory

__all__ = ["Rnn", "RnnBase"]


class RnnBase(nn.Module):
    input_size: int
    hidden_size: int

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, input: torch.Tensor, memory: Memory = None) -> tuple[torch.Tensor, Memory]:
        raise NotImplementedError


RnnLike: TypeAlias = nn.RNNBase | RnnBase


class RnnFactory(ModuleFactory["Rnn"]):
    def __init__(self, module_cls: str | type[RnnLike], **kwargs):
        if isinstance(module_cls, str):
            # RNN / LSTM / GRU
            module_cls = getattr(nn, module_cls.upper())
        self.module_cls: type[RnnLike] = module_cls
        self.kwargs = kwargs

    def __call__(self, input_dim: int, output_dim: int | None = None):
        return Rnn(self.module_cls(input_size=input_dim, **self.kwargs), output_dim)


class Rnn(Module):
    Factory = RnnFactory

    def __init__(self, rnn: RnnLike, output_dim: int | None = None):
        super().__init__(rnn.input_size, output_dim or rnn.hidden_size, is_recurrent=True)
        self.rnn = rnn
        self.output_proj = nn.Linear(rnn.hidden_size, output_dim) if output_dim else nn.Identity()

    def forward(
        self,
        input: torch.Tensor,
        *,
        memory: Memory = None,
        done: torch.Tensor | None = None,
        pack_sequence: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, Memory]:
        if done is not None:
            if pack_sequence:
                return self._forward_packed_sequence(input, memory, done)
            return self._forward_sequence(input, memory, done)
        return self._forward_tensor(input, memory)

    def _forward_tensor(
        self,
        input: torch.Tensor,
        memory: Memory = None,
    ) -> tuple[torch.Tensor, Memory]:
        if input.dim() not in (2, 3):
            raise ValueError("Input of RNNs must be 2- or 3-dimensional.")
        if input.dim() == 3:
            latent, memory = self.rnn(input, memory)
        else:
            # for x.dim() == 2, treat the 1st dim as batch instead of time
            latent, memory = self.rnn(input.unsqueeze(0), memory)
            latent = latent.squeeze(0)
        return self.output_proj(latent), memory

    def _forward_sequence(
        self,
        input: torch.Tensor,
        memory: Memory,
        done: torch.Tensor,
    ) -> tuple[torch.Tensor, Memory]:
        if input.dim() != 3:
            raise ValueError(f"Input sequences of RNNs must be 3-dimensional, got {input.ndim}.")
        padded_input, mask = split_and_pad_sequences(input, done)
        padded_latent, _ = self.rnn(padded_input, scatter_memory(memory, done))
        latent = unpad_and_merge_sequences(padded_latent, mask)
        return self.output_proj(latent), None

    def _forward_packed_sequence(
        self,
        input: torch.Tensor,
        memory: Memory,
        done: torch.Tensor,
    ) -> tuple[torch.Tensor, Memory]:
        # a slower version of forward_sequence, but preserves the final memory
        if input.dim() != 3:
            raise ValueError(f"Input of RNNs must be 3-dimensional to be packed, got {input.ndim}.")
        sequence_lengths = compute_sequence_lengths(done)
        padded_input, mask = split_and_pad_sequences(input, done)
        packed_input = nn.utils.rnn.pack_padded_sequence(
            padded_input,
            lengths=sequence_lengths.cpu(),
            enforce_sorted=False,
        )
        packed_latent, memory = self.rnn(packed_input, scatter_memory(memory, done))
        padded_latent, _ = nn.utils.rnn.pad_packed_sequence(packed_latent)
        latent = unpad_and_merge_sequences(
            padded_latent,
            mask[: padded_latent.size(0)],
            original_sequence_len=input.size(0),
        )
        memory = gather_memory(memory, done)
        return self.output_proj(latent), memory

    def step_memory(self, input: torch.Tensor, memory: Memory = None, **kwargs):
        if input.dim() not in (2, 3):
            raise ValueError("Input of RNNs must be 2- or 3-dimensional.")
        if input.dim() == 2:
            input = input.unsqueeze(0)
        _, memory = self.rnn(input, memory)
        return memory


@torch.jit.script
def get_memory_mask(done: torch.Tensor) -> torch.Tensor:
    done = done.squeeze(-1)
    last_was_done = torch.zeros_like(done, dtype=torch.bool)
    last_was_done[1:] = done[:-1]
    last_was_done[0] = True
    return last_was_done.permute(1, 0)


def scatter_memory(memory: Memory, done: torch.Tensor):
    if memory is None:
        return None
    if isinstance(memory, tuple):
        return tuple(scatter_memory(mem, done) for mem in memory)

    done = done.squeeze(-1)
    seq_indices = done[:-1].sum(dim=0).cumsum(dim=0)
    seq_indices += torch.arange(1, seq_indices.size(0) + 1, device=done.device)
    num_seq: int = seq_indices[-1].item()
    seq_indices[-1] = 0
    seq_indices = seq_indices.roll(1)

    result_shape = list(memory.shape)
    result_shape[-2] = num_seq
    result = memory.new_zeros(*result_shape)
    result[..., seq_indices, :] = memory
    return result


def gather_memory(memory: Memory, done: torch.Tensor):
    if memory is None:
        return None
    if isinstance(memory, tuple):
        return tuple(gather_memory(mem, done) for mem in memory)

    done = done.squeeze(-1)
    seq_indices = done[:-1].sum(dim=0).cumsum(dim=0)
    seq_indices += torch.arange(0, seq_indices.size(0), device=done.device)
    result = memory[..., seq_indices, :].clone()
    result[..., done[-1], :] = 0.0  # Clear the last hidden state
    return result
