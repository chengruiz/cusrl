import cusrl
from cusrl_test import create_dummy_env


def test_smoothness():
    agent_factory = cusrl.preset.ppo.RecurrentAgentFactory()
    agent_factory.register_hook(
        cusrl.hook.ActionSmoothnessLoss(0.01, 0.01),
        after="PpoSurrogateLoss",
    )
    cusrl.Trainer(create_dummy_env, agent_factory, num_iterations=5).run_training_loop()
