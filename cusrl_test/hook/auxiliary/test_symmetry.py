import pytest
import torch

import cusrl
from cusrl.utils.scheduler import StepScheduler
from cusrl_test import create_dummy_env


def test_transition_mirror_rewrites_transition_with_selected_variant():
    def stacked_self_inverse_mirror(tensor):
        return torch.stack([tensor.flip(-1), -tensor], dim=0)

    hook = cusrl.hook.TransitionMirror(index=1)
    hook.mirror_observation = stacked_self_inverse_mirror
    hook.mirror_state = stacked_self_inverse_mirror
    hook.mirror_action = stacked_self_inverse_mirror

    transition = {
        "observation": torch.tensor([[1.0, 2.0, 3.0]]),
        "state": torch.tensor([[4.0, 5.0, 6.0]]),
    }
    hook.pre_act(transition)
    torch.testing.assert_close(transition["observation"], torch.tensor([[-1.0, -2.0, -3.0]]))
    torch.testing.assert_close(transition["state"], torch.tensor([[-4.0, -5.0, -6.0]]))

    transition["action"] = torch.tensor([[7.0, 8.0, 9.0]])
    hook.post_act(transition)
    torch.testing.assert_close(transition["action"], torch.tensor([[-7.0, -8.0, -9.0]]))

    transition["next_observation"] = torch.tensor([[10.0, 11.0, 12.0]])
    transition["next_state"] = torch.tensor([[13.0, 14.0, 15.0]])
    hook.post_step(transition)
    torch.testing.assert_close(transition["next_observation"], torch.tensor([[-10.0, -11.0, -12.0]]))
    torch.testing.assert_close(transition["next_state"], torch.tensor([[-13.0, -14.0, -15.0]]))


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
