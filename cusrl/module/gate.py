import torch
from torch import Tensor, nn

__all__ = [
    "Gate",
    "GruGate",
    "HighwayGate",
    "InputGate",
    "OutputGate",
    "PassthroughGate",
    "ResidualGate",
    "SigmoidTanhGate",
    "get_gate_cls",
]


class Gate(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        raise NotImplementedError


class PassthroughGate(Gate):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return y


class ResidualGate(Gate):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return x + y


class InputGate(Gate):
    r"""
    .. math::
        g(x, y) = \sigma(W_g x) \odot x + y
    """

    def __init__(self, embed_dim: int):
        super().__init__(embed_dim)
        self.gate_linear = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        gate = torch.sigmoid(self.gate_linear(x))
        return gate * x + y


class OutputGate(Gate):
    r"""
    .. math::
        g(x, y) = x + \sigma(W_g x - b_g) \odot y
    """

    def __init__(self, embed_dim: int):
        super().__init__(embed_dim)
        self.gate_linear = nn.Linear(embed_dim, embed_dim)
        self.gate_linear.bias.data.fill_(-1.0)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        gate = torch.sigmoid(self.gate_linear(x))
        return x + gate * y


class HighwayGate(Gate):
    r"""
    .. math::
        g(x, y) = \sigma(W_g x + b_g) \odot x + (1 - \sigma(W_g x + b_g)) \odot y
    """

    def __init__(self, embed_dim: int):
        super().__init__(embed_dim)
        self.gate_linear = nn.Linear(embed_dim, embed_dim)
        self.gate_linear.bias.data.fill_(1.0)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        gate = torch.sigmoid(self.gate_linear(x))
        return gate * x + (1 - gate) * y


class SigmoidTanhGate(Gate):
    r"""
    .. math::
        g(x, y) = x + \sigma(W_g y - b_g) \odot \tanh(U_g y)$
    """

    def __init__(self, embed_dim: int):
        super().__init__(embed_dim)
        self.sigmoid_linear = nn.Linear(embed_dim, embed_dim)
        self.sigmoid_linear.bias.data.fill_(-1.0)
        self.tanh_linear = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        sigmoid_gate = torch.sigmoid(self.sigmoid_linear(y))
        tanh_activation = torch.tanh(self.tanh_linear(y))
        return x + sigmoid_gate * tanh_activation


class GruGate(Gate):
    r"""A Gated Recurrent Unit (GRU)-inspired gate.

    Described in:
    "Stabilizing Transformers for Reinforcement Learning",
    https://proceedings.mlr.press/v119/parisotto20a

    .. math::
        r = \sigma(W_r y + U_r x)                    \\
        z = \sigma(W_z y + U_z x - b_g               \\
        \hat{h} = \tanh(W_g y + U_g (r \odot x))     \\
        g(x, y) = (1 - z) \odot x + z \odot \hat{h}
    """

    def __init__(self, embed_dim: int):
        super().__init__(embed_dim)
        self.r_y = nn.Linear(embed_dim, embed_dim, bias=False)
        self.r_x = nn.Linear(embed_dim, embed_dim, bias=False)

        self.z_y = nn.Linear(embed_dim, embed_dim, bias=False)
        self.z_x = nn.Linear(embed_dim, embed_dim)
        self.z_x.bias.data.fill_(-2.0)

        self.h_y = nn.Linear(embed_dim, embed_dim, bias=False)
        self.h_rx = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # Reset gate r
        r = torch.sigmoid(self.r_y(y) + self.r_x(x))
        # Update gate z
        z = torch.sigmoid(self.z_y(y) + self.z_x(x))
        # Candidate state ĥ
        h_hat = torch.tanh(self.h_y(y) + self.h_rx(r * x))
        # Final output
        return (1 - z) * x + z * h_hat


gate_map = {
    None: PassthroughGate,
    "gru": GruGate,
    "highway": HighwayGate,
    "input": InputGate,
    "output": OutputGate,
    "residual": ResidualGate,
    "sigmoid_tanh": SigmoidTanhGate,
}


def get_gate_cls(gate_type: str | None) -> type[Gate]:
    if (gate_cls := gate_map.get(gate_type)) is None:
        raise ValueError(f"Invalid gate_type '{gate_type}'. Available: {list(gate_map.keys())}")
    return gate_cls
