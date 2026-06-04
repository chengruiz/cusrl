import pytest
import torch

import cusrl
from cusrl_test import create_dummy_env


def _create_agent_with_hook(hook: cusrl.Hook):
    agent_factory = cusrl.preset.PpoAgentFactory(
        actor_hidden_dims=(32, 16),
        critic_hidden_dims=(32, 16),
        desired_kl_divergence=None,
        device="cpu",
        autocast=False,
    ).to_underlying()
    agent_factory.register_hook(hook)
    return agent_factory.from_environment(create_dummy_env())


def _get_hook(agent: cusrl.ActorCritic, hook_type: type[cusrl.Hook]):
    return next(hook for hook in agent.hook if isinstance(hook, hook_type))


def _expected_lrs(hook):
    expected = []
    for base_lr, param_group in zip(hook._base_lrs, hook.agent.optimizer.param_groups):
        if hook.scale_all_params or any(name.startswith("actor.") for name in param_group["param_names"]):
            expected.append(base_lr * hook._lr_scale)
        else:
            expected.append(base_lr)
    return expected


def _current_lrs(agent: cusrl.ActorCritic):
    return [param_group["lr"] for param_group in agent.optimizer.param_groups]


def test_threshold_lr_schedule_rolls_back_weights_without_restoring_lr_scale():
    agent = _create_agent_with_hook(
        cusrl.hook.ThresholdLRSchedule(
            desired_kl_divergence=0.01,
            threshold=1.2,
            scale_factor=2.0,
            max_kl_divergence=0.02,
        )
    )
    hook = _get_hook(agent, cusrl.hook.ThresholdLRSchedule)
    actor_param = next(agent.actor.parameters())
    original_actor_param = actor_param.detach().clone()

    hook.pre_update(agent.buffer)
    with torch.no_grad():
        actor_param.add_(1.0)

    agent.metrics.clear()
    agent.record(kl_divergence=0.03)
    hook.post_update()

    assert torch.allclose(actor_param, original_actor_param)
    assert hook._lr_scale == pytest.approx(0.5)
    assert _current_lrs(agent) == pytest.approx(_expected_lrs(hook))
    assert agent.metrics["update_rejected"].mean.item() == pytest.approx(1.0)
