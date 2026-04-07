import pytest
import torch
from torch import nn

import cusrl


def test_actor_forward_explore_act_and_distribution_metrics():
    torch.manual_seed(0)
    actor = cusrl.Actor.Factory(
        backbone_factory=cusrl.Mlp.Factory(hidden_dims=[8], activation_fn="Tanh"),
        distribution_factory=cusrl.NormalDist.Factory(bijector="exp"),
    )(4, 2)
    observation = torch.randn(3, 4)

    dist_params, memory = actor(observation)
    det_dist_params, (det_action, det_logp), det_memory = actor.explore(observation, deterministic=True)
    action, act_memory = actor.act(observation, deterministic=True)

    assert memory is None
    assert det_memory is None
    assert act_memory is None
    assert dist_params["mean"].shape == (3, 2)
    assert dist_params["std"].shape == (3, 2)
    assert torch.allclose(det_action, det_dist_params["mean"])
    assert torch.allclose(action, det_dist_params["mean"])
    assert torch.allclose(det_logp, actor.compute_logp(det_dist_params, det_action))
    assert actor.compute_entropy(det_dist_params).shape == (3, 1)
    assert torch.allclose(actor.compute_kl_div(det_dist_params, det_dist_params), torch.zeros(3, 1), atol=1e-6)
    assert "backbone.output" in actor.intermediate_repr


def test_actor_rejects_unknown_forward_type():
    actor = cusrl.Actor.Factory(
        backbone_factory=cusrl.Mlp.Factory(hidden_dims=[8]),
        distribution_factory=cusrl.NormalDist.Factory(),
    )(4, 2)

    with pytest.raises(ValueError, match="Unsupported 'forward_type'"):
        actor(torch.randn(2, 4), forward_type="invalid")


def test_value_evaluate_matches_forward_for_state_value():
    critic = cusrl.Value.Factory(backbone_factory=cusrl.Mlp.Factory(hidden_dims=[8]))(4, 1)
    state = torch.randn(3, 4)

    value, memory = critic(state)
    evaluated = critic.evaluate(state)

    assert memory is None
    assert value.shape == (3, 1)
    assert torch.allclose(value, evaluated)
    assert "backbone.output" in critic.intermediate_repr


def test_action_aware_value_requires_action_input():
    critic = cusrl.Value(
        backbone=cusrl.Mlp(input_dim=6, hidden_dims=[8]),
        value_head=nn.Linear(8, 1),
        action_aware=True,
    )
    state = torch.randn(3, 4)
    action = torch.randn(3, 2)

    value, memory = critic(state, action=action)

    assert memory is None
    assert value.shape == (3, 1)
    with pytest.raises(ValueError, match="Action must be provided"):
        critic(state)
