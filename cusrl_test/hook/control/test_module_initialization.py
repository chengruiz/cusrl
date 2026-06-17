import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import cusrl


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 2)
        self.distribution = nn.Module()
        self.distribution.mean_head = nn.Linear(2, 1)


def test_module_initialization_orthogonalizes_actor_critic_and_zeroes_biases():
    actor = Actor()
    critic = nn.Sequential(nn.Linear(3, 2), nn.ReLU(), nn.Linear(2, 1))
    hook = cusrl.hook.ModuleInitialization(scale=math.sqrt(2), scale_dist=0.1, zero_bias=True)
    hook.agent = SimpleNamespace(actor=actor, critic=critic)

    hook.init()

    assert torch.allclose(actor.backbone.bias, torch.zeros_like(actor.backbone.bias))
    assert torch.allclose(actor.distribution.mean_head.bias, torch.zeros_like(actor.distribution.mean_head.bias))
    assert torch.allclose(critic[0].bias, torch.zeros_like(critic[0].bias))
    assert actor.distribution.mean_head.weight.norm().item() == pytest.approx(0.1)
    assert critic[0].weight[0].norm().item() == pytest.approx(math.sqrt(2))


@pytest.mark.parametrize("rnn_cls", [nn.RNN, nn.GRU, nn.LSTM])
def test_module_initialization_supports_recurrent_layers_without_bias(rnn_cls):
    module = rnn_cls(input_size=3, hidden_size=4, num_layers=2, bias=False)
    hook = cusrl.hook.ModuleInitialization()
    gate_multiplier = {nn.RNN: 1, nn.GRU: 3, nn.LSTM: 4}[rnn_cls]

    hook._init_module(module, scale=1.0, zero_bias=True)

    for layer in range(module.num_layers):
        assert getattr(module, f"weight_hh_l{layer}").shape == (4 * gate_multiplier, 4)
        assert not hasattr(module, f"bias_hh_l{layer}")
        assert not hasattr(module, f"bias_ih_l{layer}")
