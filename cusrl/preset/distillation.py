from collections.abc import Iterable
from dataclasses import dataclass

import torch

import cusrl
from cusrl.preset.optimizer import AdamFactory

__all__ = [
    "AgentFactory",
    "hook_suite",
]


def hook_suite(
    init_distribution_std: float | None = None,
    expert_path: str = "",
    expert_observation_name: str = "observation",
    normalize_observation: bool = False,
    max_grad_norm: float | None = 1.0,
) -> list[cusrl.template.Hook]:
    hooks = [
        cusrl.hook.ModuleInitialization(
            init_actor=False,
            init_critic=False,
            distribution_std=init_distribution_std,
        ),
        cusrl.hook.ObservationNormalization() if normalize_observation else None,
        cusrl.hook.OnPolicyPreparation(),
        cusrl.hook.PolicyDistillation(expert_path, expert_observation_name),
        cusrl.hook.GradientClipping(max_grad_norm) if max_grad_norm is not None else None,
    ]
    return [hook for hook in hooks if hook is not None]


@dataclass
class AgentFactory(cusrl.template.ActorCritic.Factory):
    num_steps_per_update: int = 24
    actor_hidden_dims: Iterable[int] = (256, 128)
    critic_hidden_dims: Iterable[int] = (256, 128)
    activation_fn: str | type[torch.nn.Module] = "ReLU"
    lr: float = 2e-4
    sampler_epochs: int = 1
    sampler_mini_batches: int = 8
    init_distribution_std: float | None = None
    expert_path: str = ""
    expert_observation_name: str = "observation"
    normalize_observation: bool = False
    max_grad_norm: float | None = 1.0
    device: str | torch.device | None = None
    compile: bool = False
    autocast: bool | None | str | torch.dtype = False

    def __post_init__(self):
        super().__init__(
            num_steps_per_update=self.num_steps_per_update,
            actor_factory=cusrl.Actor.Factory(
                backbone_factory=cusrl.Mlp.Factory(
                    hidden_dims=self.actor_hidden_dims,
                    activation_fn=self.activation_fn,
                    ends_with_activation=True,
                ),
                distribution_factory=cusrl.NormalDist.Factory(),
            ),
            critic_factory=cusrl.Value.Factory(
                backbone_factory=cusrl.StubModule.Factory(),
            ),
            optimizer_factory=AdamFactory(defaults={"lr": self.lr}),
            sampler=cusrl.AutoMiniBatchSampler(
                num_epochs=self.sampler_epochs,
                num_mini_batches=self.sampler_mini_batches,
            ),
            hooks=hook_suite(
                init_distribution_std=self.init_distribution_std,
                expert_path=self.expert_path,
                expert_observation_name=self.expert_observation_name,
                normalize_observation=self.normalize_observation,
                max_grad_norm=self.max_grad_norm,
            ),
            device=self.device,
            compile=self.compile,
            autocast=self.autocast,
        )
