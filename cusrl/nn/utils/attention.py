import math
from collections.abc import Callable

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import create_block_mask

__all__ = [
    "alibi_score_mod",
    "causal_sliding_window_block_mask",
    "compute_segment_ids",
    "get_alibi_slopes",
]


def get_alibi_slopes(nheads: int) -> list[float]:
    """Compute ALiBi attention bias slopes for each head.

    Implements the slope schedule from "Train Short, Test Long: Attention with
    Linear Biases Enables Input Length Extrapolation" (Press et al., 2022).

    Args:
        nheads (int): Number of attention heads.

    Returns:
        list[float]: A list of per-head slope values.
    """

    def _get_slopes_power_of_2(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * start**i for i in range(n)]

    if math.log2(nheads).is_integer():
        return _get_slopes_power_of_2(nheads)
    closest_power_of_2 = 2 ** math.floor(math.log2(nheads))
    return (
        _get_slopes_power_of_2(closest_power_of_2)
        + get_alibi_slopes(2 * closest_power_of_2)[0::2][: nheads - closest_power_of_2]
    )


def compute_segment_ids(
    done: Tensor,
    window_size: int,
) -> tuple[Tensor, Tensor]:
    """Compute segment IDs for query and key-value positions from a done tensor.

    Assigns an integer segment ID to each position so that tokens separated by
    a ``done`` boundary belong to different segments. Cache (key-value)
    positions before the current sequence always belong to segment 0.

    Args:
        done (Tensor):
            Boolean tensor of shape :math:`(L, N, 1)` indicating sequence
            terminations.
        window_size (int):
            Size of the KV cache window prepended before the current sequence.

    Returns:
        tuple[Tensor, Tensor]:
            - **q_segments** -- :math:`(N, L)` int32 segment IDs for queries.
            - **kv_segments** -- :math:`(N, W + L)` int32 segment IDs for
              key-value positions, where the first *W* entries are 0 (cache).
    """
    seq_len = done.size(0)
    batch_size = done.size(1)
    full_seq_len = seq_len + window_size

    done_t = done.squeeze(-1).transpose(0, 1)  # (N, L)
    shifted = torch.zeros_like(done_t)
    shifted[:, 1:] = done_t[:, :-1]
    q_segments = shifted.cumsum(dim=1).to(torch.int32)

    kv_segments = done.new_zeros(batch_size, full_seq_len, dtype=torch.int32)
    kv_segments[:, window_size:] = q_segments

    return q_segments, kv_segments


def causal_sliding_window_block_mask(
    kv_mask: Tensor,
    window_size: int,
    seq_len: int,
    q_segments: Tensor | None = None,
    kv_segments: Tensor | None = None,
):
    """Create a ``BlockMask`` for causal sliding-window attention.

    The mask allows each query at position *q* (0-indexed within the current
    sequence) to attend to key-value positions in the range
    ``[q, q + window_size]`` (inclusive), subject to validity and segment
    constraints.

    Args:
        kv_mask (Tensor):
            Boolean tensor of shape :math:`(N, W + L)` indicating valid
            key-value positions.
        window_size (int):
            Size of the sliding window / KV cache.
        seq_len (int):
            Length of the current query sequence.
        q_segments (Tensor | None):
            Optional :math:`(N, L)` int32 segment IDs for queries.
        kv_segments (Tensor | None):
            Optional :math:`(N, W + L)` int32 segment IDs for key-values.

    Returns:
        BlockMask: A block mask suitable for :func:`flex_attention`.
    """
    batch_size = kv_mask.size(0)
    full_seq_len = seq_len + window_size
    W = window_size
    _kv_mask = kv_mask
    _q_seg = q_segments
    _kv_seg = kv_segments

    def mask_mod(b, h, q_idx, kv_idx):
        causal = kv_idx <= q_idx + W
        window = kv_idx >= q_idx
        valid = _kv_mask[b, kv_idx]
        mask = causal & window & valid
        if _q_seg is not None:
            mask = mask & (_q_seg[b, q_idx] == _kv_seg[b, kv_idx])
        return mask

    return create_block_mask(
        mask_mod,
        B=batch_size,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=full_seq_len,
        device=kv_mask.device,
    )


def alibi_score_mod(
    alibi_slopes: Tensor,
    window_size: int,
) -> Callable:
    """Return a ``score_mod`` function that applies ALiBi bias.

    The bias subtracts ``slope * distance`` from each raw attention score,
    where *distance* is the absolute position gap between the query and
    key-value token.

    Args:
        alibi_slopes (Tensor):
            1-D tensor of shape :math:`(H,)` with per-head slopes.
        window_size (int):
            Size of the KV cache window (used to compute absolute positions).

    Returns:
        Callable: A ``score_mod(score, b, h, q_idx, kv_idx)`` function.
    """
    _alibi = alibi_slopes
    _W = window_size

    def score_mod(score, b, h, q_idx, kv_idx):
        return score - _alibi[h] * ((q_idx + _W) - kv_idx)

    return score_mod
