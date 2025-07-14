import torch
from torch import Tensor

__all__ = [
    "compute_cumulative_sequence_lengths",
    "compute_cumulative_timesteps",
    "compute_sequence_indices",
    "compute_sequence_lengths",
    "compute_reverse_cumulative_timesteps",
    "cumulate_sequence_lengths",
    "split_and_pad_sequences",
    "unpad_and_merge_sequences",
]


@torch.jit.script
def compute_sequence_indices(done: Tensor) -> Tensor:
    """Computes cumulative sequence boundary indices from end-of-sequence flags.

    This function calculates the cumulative number of sequences across different
    parallel environments based on a `done` tensor. The resulting indices are
    useful for mapping between a flat list of all sequences and their
    original environment sources.

    Args:
        done (Tensor):
            A boolean tensor of shape `(T, N, 1)` indicating sequence terminations,
            where `T` is the number of time steps and `N` is the number of parallel
            environments. If `done[t, n, 0] == 1`, it signifies the end of a sequence
            at time `t` in environment `n`.

    Returns:
        indices (Tensor): A 1D int64 tensor of shape `(N + 1,)`. The first element
            is always zero. `indices[i]` represents the total number of completed
            sequences in environments `0` through `i-1`.
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
            A boolean tensor of shape `(T, N, 1)` indicating sequence terminations.
            `T` is the number of time steps, and `N` is the number of parallel
            environments. A value of `1` at `done[t, n, 0]` signifies that the
            sequence in environment `n` ends at time step `t`.

    Returns:
        sequence_lens (Tensor):
            A 1D int64 tensor containing the length of each identified sequence.
    """
    done = done.clone()
    done[-1] = 1
    # Permute the done flag to have order (N, T, 1)
    flat_done = done.transpose(1, 0).flatten()

    # Get length of trajectory by counting the number of successive not done elements
    indices = flat_done.nonzero().squeeze(-1)
    padded_indices = indices.new_full((indices.size(0) + 1,), -1)
    padded_indices[1:] = indices

    sequence_lens = padded_indices[1:] - padded_indices[:-1]
    return sequence_lens


@torch.jit.script
def cumulate_sequence_lengths(sequence_lens: Tensor) -> Tensor:
    """Computes cumulative sequence lengths based on a sequence length tensor."""
    cumulative_sequence_lens = sequence_lens.new_zeros(sequence_lens.size(0) + 1)
    cumulative_sequence_lens[1:] = sequence_lens.cumsum(dim=0)
    return cumulative_sequence_lens


@torch.jit.script
def compute_cumulative_sequence_lengths(done: Tensor) -> Tensor:
    """Computes cumulative sequence lengths based on a 'done' tensor."""
    return cumulate_sequence_lengths(compute_sequence_lengths(done))


@torch.jit.script
def split_and_pad_sequences(compact_sequences: Tensor, done: Tensor) -> tuple[Tensor, Tensor]:
    """Splits and pads sequences from a batch of environment rollouts.

    This function processes a batch of trajectory data from multiple parallel
    environments. It extracts individual sequences based on termination flags
    and pads them to a uniform length, which is the number of timesteps in
    the input `compact_sequences`.

    Args:
        compact_sequences (Tensor):
            A tensor of compact sequence data of shape `(T, N, C)`, where `T`
            is the number of timesteps, `N` is the number of environments, and
            `C` is the channel dimension.
        done (Tensor):
            A boolean tensor of shape `(T, N, 1)` that indicates sequence
            terminations. If `done[t, n, 0] == 1`, it signifies the end of a
            sequence at time `t` in environment `n`.

    Returns:
        A tuple containing:
          - padded_sequences (Tensor):
                A tensor of shape `(T, E, C)`, where `E >= N` is the total number of
                 episodes extracted from all environments. Each episode is padded
                 with zeros to length `T`.
          - mask (Tensor):
                A boolean tensor of shape `(T, E)`. `mask[t, i]` is `True` if
                timestep `t` is a valid part of episode `i` (i.e., not padding).
    """
    if compact_sequences.dim() != 3:
        raise ValueError(f"'compact_sequences' must be 3-dimensional, but got shape {compact_sequences.shape}.")
    if done.dim() != 3 or done.size(-1) != 1:
        raise ValueError(f"'done' must be 3-dimensional with the last dimension of 1, but got shape {done.shape}.")

    max_len = compact_sequences.size(0)
    seq_lens = compute_sequence_lengths(done)
    num_seq = seq_lens.size(0)
    padded_seqs = compact_sequences.new_zeros(num_seq, max_len, compact_sequences.size(-1))
    mask = seq_lens.unsqueeze_(1) > torch.arange(0, max_len, device=seq_lens.device)
    padded_seqs[mask] = compact_sequences.transpose(0, 1).flatten(0, 1)
    return padded_seqs.transpose(0, 1), mask.transpose(0, 1)


@torch.jit.script
def unpad_and_merge_sequences(
    padded_sequences: Tensor,
    masks: Tensor,
    original_sequence_len: int | None = None,
) -> Tensor:
    """Does the inverse operation of `split_and_pad_sequences`"""
    if original_sequence_len is None:
        original_sequence_len = padded_sequences.size(0)
    return (
        padded_sequences.transpose(1, 0)[masks.transpose(1, 0)]
        .reshape(-1, original_sequence_len, padded_sequences.size(-1))
        .transpose(1, 0)
    )


@torch.jit.script
def compute_cumulative_timesteps(done: Tensor) -> Tensor:
    valid, mask = split_and_pad_sequences(torch.ones_like(done), done)
    cumulative_timesteps = valid.cumsum(dim=0) - 1
    return unpad_and_merge_sequences(cumulative_timesteps, mask)


@torch.jit.script
def compute_reverse_cumulative_timesteps(done: Tensor) -> Tensor:
    valid, mask = split_and_pad_sequences(torch.ones_like(done), done)
    cumulative_timesteps = valid.cumsum(dim=0)
    reverse_cumulative_timesteps = cumulative_timesteps[-1] - cumulative_timesteps
    return unpad_and_merge_sequences(reverse_cumulative_timesteps, mask)
