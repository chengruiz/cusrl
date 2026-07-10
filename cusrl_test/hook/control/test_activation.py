from collections.abc import Iterable

import cusrl
from cusrl_test import create_dummy_env


class DummyHook(cusrl.Hook):
    def __init__(self, epoch_index: int | Iterable[int]):
        super().__init__()
        self.epoch_index = set([epoch_index] if isinstance(epoch_index, int) else epoch_index)

    def objective(self, metadata, batch):
        assert metadata["epoch_index"] in self.epoch_index


def test_objective_activation():
    environment = create_dummy_env(with_state=True)
    agent_factory = cusrl.preset.PpoAgentFactory().to_underlying()
    agent_factory.register_hook(
        cusrl.hook.ConditionalObjectiveActivation(dummy_hook=cusrl.hook.control.EpochIndexCondition(1)),
    )
    agent_factory.register_hook(DummyHook(1))
    cusrl.Trainer(environment, agent_factory, num_iterations=1).run_training_loop()


class DummyHook2(cusrl.Hook):
    def objective(self, metadata, batch):
        assert self.agent.iteration % 2 == 0


class InitiallyInactiveHook(cusrl.Hook):
    def __init__(self):
        super().__init__()
        self.events = []
        self.active_(False)

    def pre_init(self, agent):
        super().pre_init(agent)
        self.events.append("pre_init")

    def init(self):
        self.events.append("init")

    def post_init(self):
        self.events.append("post_init")

    def pre_act(self, transition):
        self.events.append("pre_act")


def test_hook_activation():
    environment = create_dummy_env(with_state=True)
    agent_factory = cusrl.preset.PpoAgentFactory().to_underlying()
    agent_factory.register_hook(DummyHook2())
    agent_factory.register_hook(cusrl.hook.HookActivationSchedule("dummy_hook2", lambda it: it % 2 == 0))
    cusrl.Trainer(environment, agent_factory, num_iterations=5).run_training_loop()


def test_initially_inactive_hook_is_initialized_before_schedule_activation():
    environment = create_dummy_env()
    agent_factory = cusrl.preset.PpoAgentFactory().to_underlying()
    hook = InitiallyInactiveHook()
    agent_factory.register_hook(hook)
    agent_factory.register_hook(cusrl.hook.HookActivationSchedule(hook.name, lambda _iteration: True))

    agent = agent_factory.from_environment(environment)

    assert hook.active
    assert hook.agent is agent
    assert hook.events == ["pre_init", "init", "post_init"]

    observation, state, _ = environment.reset()
    agent.act(observation, state)

    assert hook.events == ["pre_init", "init", "post_init", "pre_act"]
