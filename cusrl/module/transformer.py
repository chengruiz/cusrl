from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from cusrl.module.gate import get_gate_cls
from cusrl.module.mha import MultiheadSelfAttention
from cusrl.module.module import Module, ModuleFactory

__all__ = ["FeedForward", "TransformerEncoderLayer"]


@dataclass(slots=True)
class FeedForwardFactory(ModuleFactory["FeedForward"]):
    feedforward_dim: int | None = None
    activation_fn: type[nn.Module] = nn.GELU
    dropout: float = 0.0

    def __call__(self, input_dim: int, output_dim: int | None = None):
        return FeedForward(
            input_dim=input_dim,
            feedforward_dim=self.feedforward_dim,
            activation_fn=self.activation_fn,
            dropout=self.dropout,
            output_dim=output_dim,
        )


class FeedForward(Module):
    """A feed-forward network module.

    This module implements a standard feed-forward network, typically used as a
    sub-layer in a Transformer block. It consists of two linear layers with an
    activation function and optional dropout in between.

    Args:
        input_dim (int):
            The dimension of the input tensor.
        feedforward_dim (int | None, optional):
            The dimension of the hidden layer. Defaults to ``input_dim * 4``.
        activation_fn (type[nn.Module], optional):
            The activation function to use. Defaults to :class:`nn.GELU`.
        dropout (float, optional):
            The dropout rate. Defaults to ``0.0``.
        output_dim (int, optional):
            The dimension of the output tensor. Defaults to ``input_dim``.
    """

    Factory = FeedForwardFactory

    def __init__(
        self,
        input_dim: int,
        feedforward_dim: int | None = None,
        activation_fn: type[nn.Module] = nn.GELU,
        dropout: float = 0.0,
        output_dim: int | None = None,
    ):
        super().__init__(input_dim, output_dim or input_dim)
        self.feedforward_dim = feedforward_dim or input_dim * 4

        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.feedforward_dim),
            activation_fn(),
        )
        if dropout > 0.0:
            self.layers.append(nn.Dropout(dropout))
        hidden_dim = self.layers(torch.zeros(1, self.input_dim)).size(-1)
        self.layers.append(nn.Linear(hidden_dim, self.output_dim))

    def forward(self, input: Tensor) -> Tensor:
        return self.layers(input)


@dataclass(slots=True)
class TransformerEncoderLayerFactory(ModuleFactory["TransformerEncoderLayer"]):
    embed_dim: int
    num_heads: int
    feedforward_dim: int | None = None
    activation_fn: type[nn.Module] = nn.GELU
    dropout: float = 0.0
    dtype: torch.dtype = torch.float16
    gate_type: str | None = "residual"
    layer_norm: Literal[None, "pre", "post"] = "post"

    def __call__(self, input_dim: int | None = None, output_dim: int | None = None):
        return TransformerEncoderLayer(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            feedforward_dim=self.feedforward_dim,
            activation_fn=self.activation_fn,
            dropout=self.dropout,
            dtype=self.dtype,
            gate_type=self.gate_type,
            layer_norm=self.layer_norm,
            input_dim=input_dim,
            output_dim=output_dim,
        )


class TransformerEncoderLayer(Module):
    Factory = TransformerEncoderLayerFactory

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        feedforward_dim: int | None = None,
        activation_fn: type[nn.Module] = nn.GELU,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float16,
        gate_type: str | None = "residual",
        layer_norm: Literal[None, "pre", "post"] = "post",
        input_dim: int | None = None,
        output_dim: int | None = None,
    ):
        self.embed_dim = embed_dim
        self.layer_norm = layer_norm
        gate_cls = get_gate_cls(gate_type)
        super().__init__(
            input_dim=input_dim or embed_dim,
            output_dim=output_dim or embed_dim,
            is_recurrent=False,
        )

        # modules
        if self.input_dim != self.embed_dim:
            self.in_proj = nn.Linear(self.input_dim, self.embed_dim)
        else:
            self.in_proj = nn.Identity()

        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.self_attn = MultiheadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            dtype=dtype,
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
            self.out_proj = nn.Linear(self.embed_dim, self.output_dim)
        else:
            self.out_proj = nn.Identity()

    def forward(self, input: Tensor, is_causal: bool = False) -> Tensor:
        input = self.in_proj(input)
        if self.layer_norm == "pre":
            # pre-norm: norm -> attn -> add -> norm -> ff -> add
            attn_out = self.self_attn(self.norm1(input), is_causal=is_causal)
            input = self.gate1(input, self.dropout1(attn_out))

            ff_out = self.feedforward(self.norm2(input))
            input = self.gate2(input, self.dropout2(ff_out))
        elif self.layer_norm == "post":
            # post-norm: attn -> add -> norm -> ff -> add -> norm
            attn_out = self.self_attn(input, is_causal=is_causal)
            input = self.norm1(self.gate1(input, self.dropout1(attn_out)))

            ff_out = self.feedforward(input)
            input = self.norm2(self.gate2(input, self.dropout2(ff_out)))
        else:
            # no norm: attn -> add -> ff -> add
            attn_out = self.self_attn(input, is_causal=is_causal)
            input = self.gate1(input, self.dropout1(attn_out))

            ff_out = self.feedforward(input)
            input = self.gate2(input, self.dropout2(ff_out))

        return self.out_proj(input)
