import torch
from torch import Tensor
from torch.types import Number

__all__ = [
    "apply_sequence_batch_mask",
    "compute_cumulative_sequence_lengths",
    "compute_cumulative_timesteps",
    "compute_sequence_indices",
    "compute_sequence_lengths",
    "compute_reverse_cumulative_timesteps",
    "cumulate_sequence_lengths",
    "set_sequence_batch_masked_",
    "split_and_pad_sequences",
    "unpad_and_merge_sequences",
]


def apply_sequence_batch_mask(tensor: Tensor, mask: Tensor) -> Tensor:
    """Applies a boolean mask to a tensor on sequence and batch dimensions.

    Args:
        tensor (Tensor):
            The input tensor of shape :math:`(L, ..., N, C)`, where :math:`L`
            is the sequence length, :math:`N` is the batch size and :math:`C` is
            the channel dimension.
        mask (Tensor):
            A boolean tensor of shape :math:`(L, N)` used to select elements
            from the input tensor.

    Returns:
        masked_tensor (Tensor):
            The resulting tensor of shape :math:`(..., M, C)` containing the
            masked elements, where :math:`M` is the number of ``True`` values in
            the mask.
    """
    # fmt: off
    return (
        tensor             # ( L, ..., N, C )
        .movedim(-2, 1)    # ( L, N, ..., C )
        [mask]             # ( M, ..., C )
        .movedim(0, -2)    # ( ..., M, C )
    )
    # fmt: on


def set_sequence_batch_masked_(tensor: Tensor, mask: Tensor, value: Tensor | Number) -> None:
    """In-place sets elements of a tensor based on a mask.

    Args:
        tensor (Tensor):
            The input tensor of shape :math:`(L, ..., N, C)`, where :math:`L`
            is the sequence length, :math:`N` is the batch size and :math:`C` is
            the channel dimension.
        mask (Tensor):
            A boolean tensor of shape :math:`(L, N)` used to select elements
            from the input tensor.
        value (Tensor | Number):
            The value to set, which should be broadcastable to the shape of the
            selected elements in the tensor.
    """
    if isinstance(value, Tensor):
        if value.dim() > 3:
            value = value.movedim(-2, 0)  # ( ..., M, C ) -> ( M, ..., C )
        tensor.movedim(-2, 1)[mask] = value
    else:
        tensor.movedim(-2, 1).masked_fill_(mask, value)


@torch.jit.script
def compute_sequence_indices(done: Tensor) -> Tensor:
    """Computes cumulative sequence boundary indices from end-of-sequence flags.

    This function calculates the cumulative number of sequences across different
    parallel environments based on a `done` tensor. The resulting indices are
    useful for mapping between a flat list of all sequences and their
    original environment sources.

    Args:
        done (Tensor):
            A boolean tensor of shape :math:`(L, N, 1)` indicating sequence
            terminations, where :math:`L` is the sequence length and :math:`N`
            is the batch size.

    Returns:
        indices (Tensor):
            A 1D int64 tensor of shape :math:`(N + 1,)`. The first element is
            always zero. ``indices[i]`` represents the total number of completed
            sequences in environments ``0`` through ``i-1``.
    """
    done = done.clone()
    done[-1] = 1
    indices = done.new_zeros(1 + done.size(1), dtype=torch.int64)
    indices[1:] = done.squeeze(-1).sum(dim=0).cumsum(dim=0)
    return indices


@torch.jit.script
def compute_sequence_lengths(done: Tensor) -> Tensor:
    """Computes sequence lengths from a 'done' tensor.

    This function calculates the length of each episode or trajectory segment
    from a `done` tensor, which marks the end of sequences. This is often
    used to handle variable-length episodes from parallel environments,
    especially for packing sequences for recurrent neural networks.

    Args:
        done (Tensor):
            A boolean tensor of shape :math:`(L, N, 1)` indicating sequence
            terminations, where :math:`L` is the sequence length, and :math:`N`
            is the batch size.

    Returns:
        sequence_lens (Tensor):
            A 1D int64 tensor containing the length of each identified sequence.
    """
    done = done.clone()
    done[-1] = 1
    # (L, N, 1) -> (N, L, 1) -> (N * L,)
    flat_done = done.transpose(1, 0).flatten()

    # Get length of trajectory by counting the number of successive not done elements
    indices = flat_done.nonzero().squeeze(-1)
    padded_indices = indices.new_full((indices.size(0) + 1,), -1)
    padded_indices[1:] = indices

    sequence_lens = padded_indices[1:] - padded_indices[:-1]
    return sequence_lens


@torch.jit.script
def cumulate_sequence_lengths(sequence_lens: Tensor) -> Tensor:
    """Computes cumulative sequence lengths based on sequence lengths."""
    cumulative_sequence_lens = sequence_lens.new_zeros(sequence_lens.size(0) + 1)
    cumulative_sequence_lens[1:] = sequence_lens.cumsum(dim=0)
    return cumulative_sequence_lens


@torch.jit.script
def compute_cumulative_sequence_lengths(done: Tensor) -> Tensor:
    """Computes cumulative sequence lengths based on a ``done`` tensor."""
    return cumulate_sequence_lengths(compute_sequence_lengths(done))


def split_and_pad_sequences(compact_sequences: Tensor, done: Tensor) -> tuple[Tensor, Tensor]:
    """Splits and pads sequences from a batch of environment rollouts.

    This function processes a batch of trajectory data from multiple parallel
    environments. It extracts individual sequences based on termination flags
    and pads them to a uniform length, which is the sequence length of the input
    ``compact_sequences``.

    Args:
        compact_sequences (Tensor):
            A tensor of compact sequence data of shape :math:`(L, ..., N, C)`,
            where :math:`L` is the sequence length, :math:`N` is the batch size,
            and :math:`C` is the channel dimension.
        done (Tensor):
            A boolean tensor of shape :math:`(L, N, 1)` that indicates sequence
            terminations.

    Outputs:
        - **padded_sequences** (Tensor):
            A tensor of shape :math:`(L, ..., Ns, C)`, where :math:`Ns >= N` is
            the number of extracted contiguous sequences. Each sequence is
            padded with zeros to length :math:`L`.
        - **mask** (Tensor):
            A boolean tensor of shape :math:`(L, Ns)`, indicating valid
            elements of ``padded_sequences``.
    """
    if compact_sequences.dim() < 3:
        raise ValueError(f"'compact_sequences' must be at least 3D, but got shape {compact_sequences.shape}.")
    if done.dim() != 3 or done.size(-1) != 1:
        raise ValueError(f"'done' must be 3D with the last dimension of 1, but got shape {done.shape}.")

    max_sequence_len = compact_sequences.size(0)
    sequence_lens = compute_sequence_lengths(done)
    num_sequences = sequence_lens.size(0)
    padded_sequences = compact_sequences.new_zeros(
        num_sequences,
        max_sequence_len,
        *compact_sequences.shape[1:-2],
        compact_sequences.size(-1),
    )

    mask = sequence_lens.unsqueeze(1) > torch.arange(0, max_sequence_len, device=sequence_lens.device)
    padded_sequences[mask] = compact_sequences.movedim(-2, 0).flatten(0, 1)
    return padded_sequences.movedim(0, -2), mask.transpose(0, 1)


def unpad_and_merge_sequences(
    padded_sequences: Tensor,
    mask: Tensor,
    original_sequence_len: int | None = None,
) -> Tensor:
    """Does the inverse operation of ``split_and_pad_sequences``"""
    if original_sequence_len is None:
        original_sequence_len = padded_sequences.size(0)

    # fmt: off
    return (
        padded_sequences                            # (L, ..., Ns, C)
        .movedim(-2, 0)                             # (Ns, L, ..., C)
        [mask.transpose(0, 1)]                      # (N * L, ..., C)
        .unflatten(0, (-1, original_sequence_len))  # (N, L, ..., C)
        .movedim(0, -2)                             # (L, ..., N, C)
    )
    # fmt: on


def compute_cumulative_timesteps(done: Tensor) -> Tensor:
    valid, mask = split_and_pad_sequences(torch.ones_like(done), done)
    cumulative_timesteps = valid.cumsum(dim=0) - 1
    return unpad_and_merge_sequences(cumulative_timesteps, mask)


def compute_reverse_cumulative_timesteps(done: Tensor) -> Tensor:
    valid, mask = split_and_pad_sequences(torch.ones_like(done), done)
    cumulative_timesteps = valid.cumsum(dim=0)
    reverse_cumulative_timesteps = cumulative_timesteps[-1] - cumulative_timesteps
    return unpad_and_merge_sequences(reverse_cumulative_timesteps, mask)
