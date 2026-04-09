import pytest
import torch

import cusrl
from cusrl.utils.scheduler import StepScheduler
from cusrl_test import create_dummy_env


@pytest.mark.parametrize("with_state", [False, True])
@pytest.mark.parametrize("weight", [0.0, 1.0])
def test_mirror_symmetry_loss(with_state, weight):
    environment = create_dummy_env(with_state=with_state, symmetric=True)
    agent_factory = cusrl.preset.PpoAgentFactory().to_underlying()
    agent_factory.register_hook(cusrl.hook.MirrorSymmetryLoss(weight), after="ppo_surrogate_loss")
    cusrl.Trainer(environment, agent_factory, num_iterations=5).run_training_loop()


@pytest.mark.parametrize("recurrent", [False, True])
@pytest.mark.parametrize("with_state", [False, True])
@pytest.mark.parametrize("custom_mirror_function", [False, True])
def test_symmetry_data_augmentation(recurrent, with_state, custom_mirror_function):
    environment = create_dummy_env(with_state=with_state, symmetric=True)
    agent_factory = (
        cusrl.preset.RecurrentPpoAgentFactory() if recurrent else cusrl.preset.PpoAgentFactory()
    ).to_underlying()
    if custom_mirror_function:
        hook = cusrl.hook.EnvironmentSpecOverride(
            mirror_observation=lambda obs: torch.stack([obs, obs.flip(-1)]),
            mirror_state=lambda state: torch.stack([state, state.flip(-1)]),
            mirror_action=lambda act: torch.stack([act, act.flip(-1)]),
        )
        agent_factory.register_hook(hook, index=0)
    agent_factory.register_hook(cusrl.hook.SymmetricDataAugmentation(), before="value_loss")
    cusrl.Trainer(environment, agent_factory, num_iterations=5).run_training_loop()


@pytest.mark.parametrize("with_state", [False, True])
def test_symmetric_architecture(with_state):
    environment = create_dummy_env(with_state=with_state, symmetric=True)
    agent_factory = cusrl.preset.PpoAgentFactory().to_underlying()
    agent_factory.register_hook(cusrl.hook.SymmetricArchitecture(), after="module_initialization")
    cusrl.Trainer(environment, agent_factory, num_iterations=5).run_training_loop()


def test_symmetry_loss_with_schedule():
    environment = create_dummy_env(with_state=True, symmetric=True)

    agent_factory = cusrl.preset.PpoAgentFactory().to_underlying()
    agent_factory.register_hook(cusrl.hook.MirrorSymmetryLoss(0.01), after="ppo_surrogate_loss")
    agent_factory.register_hook(
        cusrl.hook.HookParameterSchedule("mirror_symmetry_loss", "weight", StepScheduler(0.1, (3, 1.0)))
    )

    class AssertWeightHook(cusrl.TrainerHook):
        def post_update(self):
            expected = 0.1 if self.trainer.iteration + 1 < 3 else 1.0
            assert self.agent.hook["mirror_symmetry_loss"].weight == expected

    trainer = cusrl.Trainer(environment, agent_factory, num_iterations=5, hooks=[AssertWeightHook()])
    trainer.run_training_loop()
