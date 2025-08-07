import cusrl
from cusrl_test import create_dummy_env


def test_reward_shaping():
    agent_factory = cusrl.preset.ppo.AgentFactory()
    agent_factory.register_hook(
        cusrl.hook.RewardShaping(lower_bound=-1.0, upper_bound=1.0),
        before="value_computation",
    )
    cusrl.Trainer(
        create_dummy_env,
        agent_factory,
        num_iterations=5,
    ).run_training_loop()
