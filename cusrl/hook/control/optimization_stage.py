from collections.abc import Iterable

import torch

from cusrl.template.actor_critic import HookList
from cusrl.template.hook import Hook, HookComposite
from cusrl.template.optimizer import OptimizerFactory
from cusrl.utils.distributed import reduce_gradients

__all__ = ["OptimizationStage"]


class OptimizationStage(HookComposite):
    """Runs a nested optimization pass with stage-scoped hooks.

    ``OptimizationStage`` is a control hook for multi-phase updates. When the
    outer agent finishes its normal objective pass, this hook executes its
    child hooks through a second ``pre_objective -> objective -> pre_optim ->
    post_optim -> post_objective`` cycle and then applies a separate optimizer
    step. This is useful for auxiliary losses or delayed updates that should
    remain distinct from the main optimization stage. During the nested
    ``pre_objective`` call, metadata is augmented with
    ``{"optimization_stage": <stage hook name>}`` so nested control hooks can
    detect which stage is running.

    Args:
        stage_name (str):
            Suffix used to name the stage hook.
        stage_hooks (Iterable[Hook]):
            Hooks executed during the nested optimization pass..
        optimizer_factory (OptimizerFactory):
            Factory used to build the optimizer for the stage.
    """

    def __init__(
        self,
        stage_name: str,
        stage_hooks: Iterable[Hook],
        optimizer_factory: OptimizerFactory,
    ):
        self.stage_name = stage_name
        self.stage_hooks = HookList.coerce(tuple(stage_hooks))
        self.optimizer_factory = optimizer_factory
        super().__init__(self.stage_hooks)
        self.name_(f"optimization_stage_{stage_name}")

        self.optimizer: torch.optim.Optimizer
        self.grad_scaler: torch.GradScaler

    def post_init(self):
        """Instantiates the stage-local optimizer and gradient scaler."""
        self.register_stateful("optimizer", self.optimizer_factory(self.agent.named_parameters()))
        self.register_stateful(
            "grad_scaler", torch.GradScaler(device=str(self.agent.device), enabled=self.agent.autocast_enabled)
        )
        super().post_init()

    def pre_objective(self, metadata, batch):
        pass

    def objective(self, metadata, batch):
        pass

    def pre_optim(self, optimizer):
        pass

    def post_optim(self):
        pass

    def post_objective(self, metadata, batch):
        """Runs the nested objective pass and performs the stage optimizer step."""
        stage_metadata = metadata | {"optimization_stage": self.name}
        super().pre_objective(stage_metadata, batch)
        with self.agent.autocast():
            objectives = super().objective(stage_metadata, batch)
        if objectives is not None:
            loss = sum(objectives.values())
            self.optimizer.zero_grad()
            scaled_loss = self.grad_scaler.scale(loss)
            scaled_loss.backward()
            self.grad_scaler.unscale_(self.optimizer)
            reduce_gradients(self.optimizer)
            super().pre_optim(self.optimizer)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            super().post_optim()
            self.agent.record(**objectives)
        super().post_objective(stage_metadata, batch)
