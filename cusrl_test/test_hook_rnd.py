from functools import partial

import pytest

import cusrl
from cusrl_test import create_dummy_env


@pytest.mark.parametrize("with_state", [False, True])
def test_rnd(with_state):
    agent_factory = cusrl.preset.ppo.AgentFactory()
    agent_factory.register_hook(
        cusrl.hook.RandomNetworkDistillation(16, 0.1, cusrl.MLP.Factory([128, 128])),
        before="ValueComputation",
    )
    cusrl.Trainer(
        partial(create_dummy_env, with_state=with_state),
        agent_factory,
        num_iterations=5,
    ).run_training_loop()
