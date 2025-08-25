import pytest
import torch

import cusrl
from cusrl.utils.scheduler import StepFunction
from cusrl_test import create_dummy_env


@pytest.mark.parametrize("with_state", [False, True])
@pytest.mark.parametrize("weight", [0.0, 1.0])
def test_symmetry_loss(with_state, weight):
    environment = create_dummy_env(with_state=with_state, symmetric=True)
    agent_factory = cusrl.preset.ppo.AgentFactory()
    agent_factory.register_hook(cusrl.hook.SymmetryLoss(weight), after="ppo_surrogate_loss")
    cusrl.Trainer(environment, agent_factory, num_iterations=5).run_training_loop()


@pytest.mark.parametrize("recurrent", [False, True])
@pytest.mark.parametrize("with_state", [False, True])
@pytest.mark.parametrize("custom_mirror_function", [False, True])
def test_symmetry_data_augmentation(recurrent, with_state, custom_mirror_function):
    environment = create_dummy_env(with_state=with_state, symmetric=True)
    agent_factory = cusrl.preset.ppo.RecurrentAgentFactory() if recurrent else cusrl.preset.ppo.AgentFactory()
    if custom_mirror_function:
        hook = cusrl.hook.EnvironmentSpecOverride(
            mirror_observation=lambda obs: torch.cat([obs, obs.flip(-1)], dim=-2),
            mirror_state=lambda state: torch.cat([state, state.flip(-1)], dim=-2),
            mirror_action=lambda act: torch.cat([act, act.flip(-1)], dim=-2),
        )
        agent_factory.register_hook(hook, index=0)
    agent_factory.register_hook(cusrl.hook.SymmetricDataAugmentation(), before="value_loss")
    cusrl.Trainer(environment, agent_factory, num_iterations=5).run_training_loop()


@pytest.mark.parametrize("with_state", [False, True])
def test_symmetric_architecture(with_state):
    environment = create_dummy_env(with_state=with_state, symmetric=True)
    agent_factory = cusrl.preset.ppo.AgentFactory()
    agent_factory.register_hook(cusrl.hook.SymmetricArchitecture(), after="module_initialization")
    cusrl.Trainer(environment, agent_factory, num_iterations=5).run_training_loop()


def test_symmetry_loss_with_schedule():
    environment = create_dummy_env(with_state=True, symmetric=True)

    agent_factory = cusrl.preset.ppo.AgentFactory()
    agent_factory.register_hook(cusrl.hook.SymmetryLoss(0.01), after="ppo_surrogate_loss")
    agent_factory.register_hook(
        cusrl.hook.HookParameterSchedule("symmetry_loss", "weight", StepFunction(0.1, (3, 1.0)))
    )

    def assert_weight_equals(trainer):
        assert trainer.agent.hook["symmetry_loss"].weight == 0.1 if trainer.iteration + 1 < 3 else 1.0

    trainer = cusrl.Trainer(environment, agent_factory, num_iterations=5)
    trainer.register_callback(assert_weight_equals)
    trainer.run_training_loop()
