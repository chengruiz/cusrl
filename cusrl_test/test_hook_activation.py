from collections.abc import Iterable
from typing import Any

import torch

import cusrl
from cusrl_test import create_dummy_env


class DummyHook(cusrl.Hook):
    def __init__(self, epoch_index: int | Iterable[int]):
        self.epoch_index = set([epoch_index] if isinstance(epoch_index, int) else epoch_index)

    def objective(self, batch: dict[str, Any]) -> torch.Tensor | None:
        assert batch["epoch_index"] in self.epoch_index


def test_objective_activation():
    environment = create_dummy_env(with_state=True)
    agent_factory = cusrl.preset.ppo.AgentFactory()
    agent_factory.register_hook(
        cusrl.hook.ConditionalObjectiveActivation(DummyHook=cusrl.hook.condition.EpochIndexCondition(1)),
    )
    agent_factory.register_hook(DummyHook(1))
    cusrl.Trainer(environment, agent_factory, num_iterations=1).run_training_loop()


class DummyHook2(cusrl.Hook):
    def objective(self, batch: dict[str, Any]) -> torch.Tensor | None:
        assert self.agent.iteration % 2 == 0


def test_hook_activation():
    environment = create_dummy_env(with_state=True)
    agent_factory = cusrl.preset.ppo.AgentFactory()
    agent_factory.register_hook(DummyHook2())
    agent_factory.register_hook(cusrl.hook.HookActivationSchedule("DummyHook2", lambda it: it % 2 == 0))
    cusrl.Trainer(environment, agent_factory, num_iterations=5).run_training_loop()
