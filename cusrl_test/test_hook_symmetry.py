import pytest

import cusrl
from cusrl_test import create_dummy_env


@pytest.mark.parametrize("with_state", [False, True])
@pytest.mark.parametrize("weight", [0.0, 1.0])
def test_symmetry_loss(with_state, weight):
    environment = create_dummy_env(with_state=with_state, symmetric=True)
    agent_factory = cusrl.preset.ppo.AgentFactory()
    agent_factory.register_hook(cusrl.hook.SymmetryLoss(weight), after="PpoSurrogateLoss")
    cusrl.Trainer(environment, agent_factory, num_iterations=5).run_training_loop()


@pytest.mark.parametrize("with_state", [False, True])
def test_symmetry_data_augmentation(with_state):
    environment = create_dummy_env(with_state=with_state, symmetric=True)
    agent_factory = cusrl.preset.ppo.AgentFactory()
    agent_factory.register_hook(cusrl.hook.SymmetricDataAugmentation(), after="OnPolicyPreparation")
    cusrl.Trainer(environment, agent_factory, num_iterations=5).run_training_loop()


@pytest.mark.parametrize("with_state", [False, True])
def test_symmetric_architecture(with_state):
    environment = create_dummy_env(with_state=with_state, symmetric=True)
    agent_factory = cusrl.preset.ppo.AgentFactory()
    agent_factory.register_hook(cusrl.hook.SymmetricArchitecture(), after="ModuleInitialization")
    cusrl.Trainer(environment, agent_factory, num_iterations=5).run_training_loop()


def test_symmetry_loss_with_schedule():
    environment = create_dummy_env(with_state=True, symmetric=True)

    agent_factory = cusrl.preset.ppo.AgentFactory()
    agent_factory.register_hook(cusrl.hook.SymmetryLoss(0.01), after="PpoSurrogateLoss")
    agent_factory.register_hook(
        cusrl.hook.ParameterSchedule("SymmetryLoss", "weight", cusrl.hook.schedule.PiecewiseFunction(0.1, (3, 1.0)))
    )

    def assert_weight_equals(trainer):
        assert trainer.agent.hook["SymmetryLoss"].weight == 0.1 if trainer.iteration + 1 < 3 else 1.0

    trainer = cusrl.Trainer(environment, agent_factory, num_iterations=5)
    trainer.register_callback(assert_weight_equals)
    trainer.run_training_loop()
