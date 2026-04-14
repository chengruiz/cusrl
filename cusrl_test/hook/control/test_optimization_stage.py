from collections.abc import Iterable

import torch

import cusrl
from cusrl.utils import from_dict, to_dict
from cusrl_test import create_dummy_env


class DummyOptimizationStageHook(cusrl.Hook[cusrl.ActorCritic]):
    def __init__(self, epoch_index: int | Iterable[int]):
        super().__init__()
        self.epoch_index = set([epoch_index] if isinstance(epoch_index, int) else epoch_index)

    def init(self):
        self.register_module("probe", torch.nn.Linear(1, 1, bias=False))

    def objective(self, metadata, batch):
        assert metadata["epoch_index"] in self.epoch_index
        return {"dummy_stage_loss": self.probe.weight.square().sum()}


def test_optimization_stage_supports_nested_control_hooks_and_checkpoint_state():
    environment = create_dummy_env(with_state=True)
    agent_factory = cusrl.preset.PpoAgentFactory().to_underlying()
    agent_factory.register_hook(
        cusrl.hook.ConditionalObjectiveActivation(
            **{"optimization_stage_aux.dummy_stage_hook": cusrl.hook.control.EpochIndexCondition(1)}
        )
    )
    agent_factory.register_hook(
        cusrl.hook.OptimizationStage(
            "aux",
            [DummyOptimizationStageHook(1).name_("dummy_stage_hook")],
            cusrl.OptimizerFactory("Adam", defaults={"lr": 1e-3}),
        )
    )
    agent_factory.register_hook(
        cusrl.hook.HookActivationSchedule("optimization_stage_aux.dummy_stage_hook", lambda it: True)
    )

    trainer = cusrl.Trainer(environment, agent_factory, num_iterations=1, verbose=False)
    trainer.run_training_loop()

    stage_state_dict = trainer.agent.state_dict()["hook"]["optimization_stage_aux"]

    assert {"dummy_stage_hook", "grad_scaler", "optimizer"} <= set(stage_state_dict)
    assert trainer.agent.hook["optimization_stage_aux.dummy_stage_hook"].name == "dummy_stage_hook"


def test_optimization_stage_round_trip_supports_iterable_stage_hooks():
    stage = cusrl.hook.OptimizationStage(
        "aux",
        (hook for hook in [cusrl.hook.GradientClipping()]),
        cusrl.OptimizerFactory("Adam", defaults={"lr": 1e-3}),
    )

    restored = from_dict(None, to_dict(stage))

    assert isinstance(stage.stage_hooks, cusrl.template.actor_critic.HookList)
    assert isinstance(restored, cusrl.hook.OptimizationStage)
    assert isinstance(restored.stage_hooks, cusrl.template.actor_critic.HookList)
    assert [hook.name for hook in restored.stage_hooks] == ["gradient_clipping"]
