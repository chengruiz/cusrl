import pytest

import cusrl
from cusrl_test import create_dummy_env


@pytest.mark.parametrize(
    "agent_factory",
    [
        cusrl.preset.ppo.AgentFactory(),
        cusrl.preset.ppo.RecurrentAgentFactory(),
    ],
)
@pytest.mark.parametrize(
    "hook",
    [
        cusrl.hook.ReturnPrediction(),
        cusrl.hook.StatePrediction(slice(16, 24)),
        cusrl.hook.NextStatePrediction(slice(16, 24)),
    ],
)
def test_representation_hook(agent_factory, hook):
    environment = create_dummy_env(with_state=True)
    agent_factory.register_hook(hook, after="PpoSurrogateLoss")
    cusrl.Trainer(environment, agent_factory, num_iterations=1).run_training_loop()
