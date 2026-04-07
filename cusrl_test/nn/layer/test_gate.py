import pytest
import torch

from cusrl.nn.layer import (
    GruGate,
    HighwayGate,
    InputGate,
    OutputGate,
    PassthroughGate,
    ResidualGate,
    SigmoidTanhGate,
    get_gate_cls,
)


def test_passthrough_and_residual_gates_match_expected_outputs():
    x = torch.tensor([[1.0, 2.0]])
    y = torch.tensor([[3.0, 4.0]])

    assert torch.allclose(PassthroughGate(2)(x, y), y)
    assert torch.allclose(ResidualGate(2)(x, y), x + y)


def test_input_gate_uses_sigmoid_weighted_input():
    gate = InputGate(2)
    gate.gate_linear.weight.data.zero_()
    x = torch.tensor([[2.0, -4.0]])
    y = torch.tensor([[1.0, 3.0]])

    output = gate(x, y)

    assert torch.allclose(output, 0.5 * x + y)


def test_output_gate_uses_sigmoid_weighted_residual_branch():
    gate = OutputGate(2)
    gate.gate_linear.weight.data.zero_()
    gate.gate_linear.bias.data.zero_()
    x = torch.tensor([[2.0, -4.0]])
    y = torch.tensor([[1.0, 3.0]])

    output = gate(x, y)

    assert torch.allclose(output, x + 0.5 * y)


def test_highway_gate_interpolates_between_input_and_residual():
    gate = HighwayGate(2)
    gate.gate_linear.weight.data.zero_()
    gate.gate_linear.bias.data.zero_()
    x = torch.tensor([[2.0, -4.0]])
    y = torch.tensor([[1.0, 3.0]])

    output = gate(x, y)

    assert torch.allclose(output, 0.5 * x + 0.5 * y)


def test_sigmoid_tanh_gate_combines_sigmoid_and_tanh_paths():
    gate = SigmoidTanhGate(2)
    gate.sigmoid_linear.weight.data.zero_()
    gate.sigmoid_linear.bias.data.zero_()
    gate.tanh_linear.weight.data.copy_(torch.eye(2))
    x = torch.tensor([[2.0, -4.0]])
    y = torch.tensor([[1.0, 0.5]])

    output = gate(x, y)

    assert torch.allclose(output, x + 0.5 * torch.tanh(y))


def test_gru_gate_reduces_to_half_input_with_zeroed_weights():
    gate = GruGate(2)
    for param in gate.parameters():
        param.data.zero_()
    x = torch.tensor([[2.0, -4.0]])
    y = torch.tensor([[1.0, 3.0]])

    output = gate(x, y)

    assert torch.allclose(output, 0.5 * x)


@pytest.mark.parametrize(
    ("gate_type", "expected_cls"),
    [
        (None, PassthroughGate),
        ("gru", GruGate),
        ("highway", HighwayGate),
        ("input", InputGate),
        ("output", OutputGate),
        ("residual", ResidualGate),
        ("sigmoid_tanh", SigmoidTanhGate),
    ],
)
def test_get_gate_cls_returns_expected_gate_class(gate_type, expected_cls):
    assert get_gate_cls(gate_type) is expected_cls


def test_get_gate_cls_rejects_unknown_gate_types():
    with pytest.raises(ValueError, match="Unsupported gate type"):
        get_gate_cls("unknown")
