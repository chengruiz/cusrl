from functools import partial

import pytest

import cusrl
from cusrl.preset import ppo
from cusrl_test import create_dummy_env


@pytest.mark.parametrize("with_state", [False, True])
@pytest.mark.parametrize(
    "agent_factory_cls",
    [
        partial(
            ppo.AgentFactory,
            actor_hidden_dims=(32, 16),
            critic_hidden_dims=(32, 16),
        ),
        partial(
            ppo.RecurrentAgentFactory,
            actor_num_layers=1,
            actor_hidden_size=32,
            critic_num_layers=1,
            critic_hidden_size=32,
        ),
    ],
)
@pytest.mark.parametrize("popart_alpha", [None, 0.01])
@pytest.mark.parametrize("gae_lamda_value", [None, 0.995])
@pytest.mark.parametrize("max_grad_norm", [None, 1.0])
@pytest.mark.parametrize("desired_kl_divergence", [None, 0.01])
@pytest.mark.parametrize("autocast", [False, True] if cusrl.utils.is_autocast_available() else [False])
def test_ppo_options(
    with_state,
    agent_factory_cls,
    popart_alpha,
    gae_lamda_value,
    max_grad_norm,
    desired_kl_divergence,
    autocast,
):
    environment = create_dummy_env(with_state=with_state)
    agent_factory = agent_factory_cls(
        popart_alpha=popart_alpha,
        gae_lamda_value=gae_lamda_value,
        max_grad_norm=max_grad_norm,
        desired_kl_divergence=desired_kl_divergence,
        autocast=autocast,
    )
    trainer = cusrl.Trainer(environment, agent_factory, num_iterations=3)
    trainer.run_training_loop()
