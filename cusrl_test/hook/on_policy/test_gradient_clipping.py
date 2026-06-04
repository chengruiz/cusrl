import pytest
import torch
from torch import nn

import cusrl
from cusrl.template.optimizer import OptimizerFactory


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Linear(2, 2)
        self.critic = nn.Linear(2, 1)


class DummyAgent:
    def __init__(self):
        self.records = {}

    def record(self, **kwargs):
        self.records.update(kwargs)


def total_grad_norm(parameters):
    grads = [parameter.grad.reshape(-1) for parameter in parameters if parameter.grad is not None]
    return torch.cat(grads).norm() if grads else torch.tensor(0.0)


def test_gradient_clipping_clips_default_and_prefixed_groups():
    model = ToyModel()
    optimizer = OptimizerFactory(
        "SGD",
        defaults={"lr": 0.1},
        param_groups={"actor": {"lr": 0.01}},
    )(model.named_parameters())
    hook = cusrl.hook.GradientClipping(max_grad_norm=0.5, actor=0.25)
    hook.agent = DummyAgent()

    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    hook.pre_optim(optimizer)

    actor_norm = total_grad_norm(list(model.actor.parameters())).item()
    critic_norm = total_grad_norm(list(model.critic.parameters())).item()

    assert actor_norm <= 0.25 + 1e-6
    assert critic_norm <= 0.5 + 1e-6
    assert set(hook.agent.records) == {"grad_norm/actor", "grad_norm/default"}


def test_gradient_clipping_rejects_negative_default_limit():
    with pytest.raises(ValueError, match="'max_grad_norm' must be non-negative"):
        cusrl.hook.GradientClipping(max_grad_norm=-1.0)
