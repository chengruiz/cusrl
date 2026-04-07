from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn.attention.flex_attention import flex_attention

from cusrl.nn.layer.encoding import apply_rotary_emb
from cusrl.nn.layer.gate import get_gate_cls
from cusrl.nn.layer.transformer import FeedForward
from cusrl.nn.module.module import Module, ModuleFactory
from cusrl.nn.utils.attention import (
    alibi_score_mod,
    causal_sliding_window_block_mask,
    compute_segment_ids,
    get_alibi_slopes,
)
from cusrl.nn.utils.recurrent import (
    compute_reverse_cumulative_timesteps,
    select_initial_memory,
)
from cusrl.utils.typing import Memory, Slice

__all__ = ["CausalMultiheadSelfAttention", "CausalTransformerEncoderLayer", "FeedForward"]


@dataclass(slots=True)
class CausalMultiheadSelfAttentionFactory(ModuleFactory["CausalMultiheadSelfAttention"]):
    embed_dim: int
    num_heads: int
    window_size: int
    dtype: torch.dtype = torch.float16
    alibi_slopes: Tensor | None = None
    rope_base: float | None = None

    def __call__(self, input_dim: int | None = None, output_dim: int | None = None):
        return CausalMultiheadSelfAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            window_size=self.window_size,
            dtype=self.dtype,
            alibi_slopes=self.alibi_slopes,
            rope_base=self.rope_base,
            input_dim=input_dim,
            output_dim=output_dim,
        )


class CausalMultiheadSelfAttention(Module):
    Factory = CausalMultiheadSelfAttentionFactory

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int,
        dtype: torch.dtype = torch.float16,
        alibi_slopes: Tensor | None = None,
        rope_base: float | None = None,
        input_dim: int | None = None,
        output_dim: int | None = None,
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.dtype = dtype
        self.alibi_slopes = torch.as_tensor(alibi_slopes) if alibi_slopes is not None else None
        self.rope_base = rope_base  # Rotary Positional Embedding
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError("'embed_dim' must be divisible by 'num_heads'")
        if self.rope_base is not None:
            if self.rope_base <= 0:
                raise ValueError("'rope_base' must be a positive number")
            if self.head_dim // 2 == 0:
                raise ValueError("'head_dim' must be even when RoPE is enabled")
        if self.alibi_slopes is not None and self.alibi_slopes.ndim != 1:
            raise ValueError("'alibi_slopes' must be a 1D tensor")
        if self.alibi_slopes is not None and self.alibi_slopes.size(0) != num_heads:
            raise ValueError(f"'alibi_slopes' must contain {num_heads} elements")
        if self.window_size <= 0:
            raise ValueError("'window_size' must be a positive integer")
        super().__init__(
            input_dim=input_dim or embed_dim,
            output_dim=output_dim or embed_dim,
            is_recurrent=True,
        )

        # projections
        self.q_proj = nn.Linear(self.input_dim, embed_dim)
        self.kv_proj = nn.Linear(self.input_dim, embed_dim * 2)
        self.out_proj = nn.Linear(embed_dim, self.output_dim)
        self._rotary_cos = self._rotary_sin = None

    def forward(
        self,
        input: Tensor,
        memory: Memory = None,
        *,
        done: Tensor | None = None,
        sequential: bool = True,
        **kwargs,
    ) -> tuple[Tensor, Memory]:
        """Computes multi-head self-attention with KV caching.

        Args:
            input (Tensor):
                Input tensor of shape :math:`(L, N, ..., C)`, where :math:`L` is
                the sequence length, :math:`N` is the batch size, and :math:`C`
                is the input dimension.
            memory (Memory):
                A dict containing the input cache and cache mask.
                  - input_cache (Tensor):
                      Tensor of shape :math:`(N, ..., W * C)` storing past
                      inputs, where :math:`W` is the window size.
                  - cache_mask (Tensor):
                      Boolean tensor of shape :math:`(N, ..., W)` indicating
                      valid cache entries.
            done (Tensor | None):
                A boolean tensor of shape :math:`(L, N, 1)` indicating sequence
                terminations.
            sequential (bool):
                If ``True``, the input is treated as a sequences. Otherwise,
                it's treated as a single batch of data. Defaults to ``True``.

        Outputs:
            - **output** (Tensor):
                The attention output tensor of the same shape as ``input``.
            - **memory** (Memory):
                The updated memory dict with ``input_cache`` and ``cache_mask``
                entries.
        """
        if seq_missing := (input.dim() == 2 or not sequential):
            input = input.unsqueeze(0)
        if sequential and input.dim() >= 3:
            memory = select_initial_memory(memory, input.shape[:-1])
        batch_dims = input.shape[1:-1]
        input = input.flatten(1, -2)

        # Convert inputs to batch first
        input = input.transpose(0, 1)
        batch_size, seq_len, _ = input.shape
        full_seq_len = seq_len + self.window_size

        # Compute query linear projections
        q = self.q_proj(input).view(batch_size, seq_len, self.num_heads, self.head_dim)

        if memory is None:
            # Initialize input cache and mask if no memory is provided
            input_cache = input.new_zeros(batch_size, self.window_size, self.input_dim)
            kv_mask = q.new_zeros(batch_size, full_seq_len, dtype=torch.bool)
            kv_mask[:, -seq_len:] = True
        else:
            input_cache = memory["input_cache"].reshape(batch_size, self.window_size, self.input_dim)
            cache_mask = memory["cache_mask"].reshape(batch_size, self.window_size)
            kv_mask = cache_mask.new_ones(batch_size, full_seq_len)
            kv_mask[:, : self.window_size] = cache_mask

        full_input = torch.cat([input_cache, input], dim=1)
        kv = torch.cat([self.kv_proj(input_cache).detach(), self.kv_proj(input)], dim=1)
        kv = kv.unflatten(-1, (2, self.num_heads, self.head_dim))

        # Expand done for multi-dimensional batch
        if done is not None and len(batch_dims) > 1:
            done = done.repeat_interleave(batch_size // batch_dims[0], dim=1)

        # Compute segment IDs for sub-sequence isolation
        if done is not None:
            q_segments, kv_segments = compute_segment_ids(done, self.window_size)
        else:
            q_segments = kv_segments = None

        # Apply RoPE
        if self.alibi_slopes is not None:
            self.alibi_slopes = self.alibi_slopes.to(device=input.device)
        if self.rope_base is not None:
            self._update_cos_sin_cache(full_seq_len, q.device)
            q = apply_rotary_emb(
                q,
                self._rotary_cos[self.window_size : self.window_size + seq_len],
                self._rotary_sin[self.window_size : self.window_size + seq_len],
            )
            k_rotated = apply_rotary_emb(
                kv[:, :, 0],
                self._rotary_cos[:full_seq_len],
                self._rotary_sin[:full_seq_len],
            )
            kv = torch.stack([k_rotated, kv[:, :, 1]], dim=2)

        # Separate K, V and transpose to (N, H, L, D)
        k, v = kv[:, :, 0], kv[:, :, 1]
        q_fa = q.transpose(1, 2).to(self.dtype)
        k_fa = k.transpose(1, 2).to(self.dtype)
        v_fa = v.transpose(1, 2).to(self.dtype)

        # Build flex_attention block mask and optional ALiBi score modifier
        block_mask = causal_sliding_window_block_mask(kv_mask, self.window_size, seq_len, q_segments, kv_segments)
        score_mod = alibi_score_mod(self.alibi_slopes, self.window_size) if self.alibi_slopes is not None else None

        attn_out = flex_attention(q_fa, k_fa, v_fa, score_mod=score_mod, block_mask=block_mask)
        attn_out = attn_out.transpose(1, 2).type_as(input)  # (N, S, H, D)

        # Combine heads and project to output_dim
        output = self.out_proj(attn_out.flatten(-2))

        # Prepare new cache tensors
        new_input_cache = full_input[:, -self.window_size :]
        new_cache_mask = kv_mask[:, -self.window_size :]

        # Restore outputs to sequence first ( L, N, * )
        output = output.transpose(0, 1)

        # Update cache mask based on done tensor
        if done is not None:
            if done.size(0) < self.window_size:
                padded_done = done.new_zeros(self.window_size, done.size(1), 1)
                padded_done[-done.size(0) :] = done
                done = padded_done
            elif done.size(0) > self.window_size:
                done = done[-self.window_size :]

            cum_timesteps = compute_reverse_cumulative_timesteps(done).squeeze(-1)
            consecutive_timesteps = torch.arange(self.window_size - 1, -1, -1, device=done.device)
            valid_cache_mask = cum_timesteps.transpose(0, 1) == consecutive_timesteps.unsqueeze(0)
            new_cache_mask = new_cache_mask.logical_and(valid_cache_mask)

        output = output.unflatten(1, batch_dims)
        if seq_missing:
            output = output.squeeze(0)
        return output, {
            "input_cache": new_input_cache.reshape(*batch_dims, self.window_size * self.input_dim),
            "cache_mask": new_cache_mask.reshape(*batch_dims, self.window_size),
        }

    def reset_memory(
        self,
        memory: dict[str, Tensor] | None,
        done: Slice | Tensor | None = None,
    ):
        """Resets the memory cache for specific environments.

        This method selectively resets the memory components (input cache and
        cache mask). If ``done`` is not provided, the entire memory is cleared.
        Otherwise, only the memory states corresponding to the ``done`` indices
        (e.g., for environments that are done) are reset.

        Args:
            memory (dict[str, Tensor] | None):
                A dict containing the input cache and cache mask. If ``None``,
                the function does nothing.
            done (SliceType | Tensor | None, optional):
                A mask or slice indicating which parts of the memory to reset.
                If it's a tensor, it should be of shape :math:`(N, 1)`. If
                ``None``, the entire memory is reset. Defaults to ``None``.
        """
        if memory is None:
            return
        input_cache = memory["input_cache"]
        cache_mask = memory["cache_mask"]
        if done is None:
            input_cache.zero_()
            cache_mask.fill_(False)
        else:
            if isinstance(done, Tensor):
                done = done.squeeze(-1)
            input_cache[done] = 0.0
            cache_mask[done] = False

    def _update_cos_sin_cache(self, seq_len, device):
        if self._rotary_sin is not None and self._rotary_sin.size(0) >= seq_len:
            return

        t = torch.arange(0.0, seq_len, device=device)
        inv_freq = 1.0 / (self.rope_base ** (torch.arange(0.0, self.head_dim, 2.0, device=device) / self.head_dim))
        freq = torch.outer(t, inv_freq)
        self._rotary_cos = freq.cos()
        self._rotary_sin = freq.sin()


@dataclass(slots=True)
class CausalTransformerEncoderLayerFactory(ModuleFactory["CausalTransformerEncoderLayer"]):
    embed_dim: int
    num_heads: int
    window_size: int
    feedforward_dim: int | None = None
    activation_fn: type[nn.Module] = nn.GELU
    dropout: float = 0.0
    dtype: torch.dtype = torch.float16
    gate_type: str | None = "residual"
    layer_norm: Literal[None, "pre", "post"] = "post"
    use_alibi: bool = False
    rope_base: float | None = None

    def __call__(self, input_dim: int | None = None, output_dim: int | None = None):
        return CausalTransformerEncoderLayer(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            window_size=self.window_size,
            feedforward_dim=self.feedforward_dim,
            activation_fn=self.activation_fn,
            dropout=self.dropout,
            dtype=self.dtype,
            gate_type=self.gate_type,
            layer_norm=self.layer_norm,
            use_alibi=self.use_alibi,
            rope_base=self.rope_base,
            input_dim=input_dim,
            output_dim=output_dim,
        )


class CausalTransformerEncoderLayer(Module):
    Factory = CausalTransformerEncoderLayerFactory

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int,
        feedforward_dim: int | None = None,
        activation_fn: type[nn.Module] = nn.GELU,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float16,
        gate_type: str | None = "residual",
        layer_norm: Literal[None, "pre", "post"] = "post",
        use_alibi: bool = False,
        rope_base: float | None = None,
        input_dim: int | None = None,
        output_dim: int | None = None,
    ):
        self.embed_dim = embed_dim
        self.layer_norm = layer_norm
        gate_cls = get_gate_cls(gate_type)
        super().__init__(
            input_dim=input_dim or embed_dim,
            output_dim=output_dim or embed_dim,
            is_recurrent=True,
        )

        # modules
        if self.input_dim != self.embed_dim:
            self.in_proj: nn.Module = nn.Linear(self.input_dim, self.embed_dim)
        else:
            self.in_proj = nn.Identity()

        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.self_attn = CausalMultiheadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            dtype=dtype,
            input_dim=embed_dim,
            output_dim=self.embed_dim,
            alibi_slopes=get_alibi_slopes(num_heads) if use_alibi else None,
            rope_base=rope_base,
        )
        self.dropout1 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.gate1 = gate_cls(self.embed_dim)

        self.norm2 = nn.LayerNorm(self.embed_dim)
        self.feedforward = FeedForward(
            input_dim=self.embed_dim,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
            output_dim=self.embed_dim,
            activation_fn=activation_fn,
        )
        self.dropout2 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.gate2 = gate_cls(self.embed_dim)

        if self.output_dim != self.embed_dim:
            self.out_proj: nn.Module = nn.Linear(self.embed_dim, self.output_dim)
        else:
            self.out_proj = nn.Identity()

    def forward(
        self,
        input: Tensor,
        memory: Memory = None,
        *,
        done: Tensor | None = None,
        sequential: bool = True,
        **kwargs,
    ) -> tuple[Tensor, Memory]:
        input = self.in_proj(input)
        if self.layer_norm == "pre":
            # pre-norm: norm -> attn -> add -> norm -> ff -> add
            attn_out, memory = self.self_attn(self.norm1(input), memory, done=done, sequential=sequential)
            input = self.gate1(input, self.dropout1(attn_out))

            ff_out = self.feedforward(self.norm2(input))
            input = self.gate2(input, self.dropout2(ff_out))
        elif self.layer_norm == "post":
            # post-norm: attn -> add -> norm -> ff -> add -> norm
            attn_out, memory = self.self_attn(input, memory, done=done, sequential=sequential)
            input = self.norm1(self.gate1(input, self.dropout1(attn_out)))

            ff_out = self.feedforward(input)
            input = self.norm2(self.gate2(input, self.dropout2(ff_out)))
        else:
            # no norm: attn -> add -> ff -> add
            attn_out, memory = self.self_attn(input, memory, done=done, sequential=sequential)
            input = self.gate1(input, self.dropout1(attn_out))

            ff_out = self.feedforward(input)
            input = self.gate2(input, self.dropout2(ff_out))

        return self.out_proj(input), memory

    def step_memory(self, input, memory=None, **kwargs):
        input = self.in_proj(input)
        if self.layer_norm == "pre":
            input = self.norm1(input)
        return self.self_attn.step_memory(input, memory, **kwargs)

    def reset_memory(
        self,
        memory: dict[str, Tensor],
        done: Slice | Tensor | None = None,
    ):
        self.self_attn.reset_memory(memory, done)
