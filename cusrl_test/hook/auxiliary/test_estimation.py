import pytest

import cusrl
from cusrl_test import create_dummy_env


@pytest.mark.parametrize(
    "estimator_factory",
    [
        cusrl.Mlp.Factory(hidden_dims=[32]),
        cusrl.Lstm.Factory(hidden_size=32),
    ],
)
def test_estimation(estimator_factory):
    environment = create_dummy_env(with_state=True)
    agent_factory = cusrl.preset.RecurrentPpoAgentFactory().to_underlying()
    agent_factory.register_hook(
        cusrl.hook.StateEstimation(
            estimator_factory=estimator_factory,
        )
    )
    cusrl.Trainer(environment, agent_factory, num_iterations=5).run_training_loop()
