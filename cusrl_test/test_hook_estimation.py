import cusrl
from cusrl_test import create_dummy_env


def test_estimation():
    environment = create_dummy_env(with_state=True)
    agent_factory = cusrl.preset.ppo.RecurrentAgentFactory()
    agent_factory.register_hook(
        cusrl.hook.StateEstimation(
            estimator_factory=cusrl.Lstm.Factory(hidden_size=32),
        )
    )
    cusrl.Trainer(environment, agent_factory, num_iterations=5).run_training_loop()
