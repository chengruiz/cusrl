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
