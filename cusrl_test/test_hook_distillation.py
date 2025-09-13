import os
from datetime import datetime

import cusrl
from cusrl_test import create_dummy_env


def test_distillation():
    env1 = create_dummy_env(observation_dim=24)
    env2 = create_dummy_env(with_state=True)

    expert = cusrl.preset.ppo.AgentFactory().from_environment(env1)
    dirname = f"/tmp/cusrl/distillation/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    expert.export(dirname, target_format="jit", batch_size=env1.num_instances)

    agent_factory = cusrl.preset.distillation.AgentFactory(
        expert_path=os.path.join(dirname, "actor.pt"),
        expert_observation_name="state",
    )
    cusrl.Trainer(env2, agent_factory, num_iterations=5).run_training_loop()
