from collections.abc import Sequence
from dataclasses import dataclass

import torch

import cusrl
from cusrl.preset.optimizer import AdamFactory
from cusrl.template.actor_critic import ActorCritic, ActorCriticFactory
from cusrl.template.agent import AgentFactory
from cusrl.template.environment import EnvironmentSpec

__all__ = [
    "DistillationAgentFactory",
    "distillation_hook_suite",
]


def distillation_hook_suite(
    expert_path: str = "",
    expert_observation_name: str = "observation",
    normalize_observation: bool = False,
    max_grad_norm: float | None = 1.0,
) -> list[cusrl.template.Hook]:
    hooks = [
        cusrl.hook.ObservationNormalization() if normalize_observation else None,
        cusrl.hook.OnPolicyPreparation(),
        cusrl.hook.PolicyDistillation(expert_path, expert_observation_name),
        cusrl.hook.GradientClipping(max_grad_norm),
    ]
    return [hook for hook in hooks if hook is not None]


@dataclass(kw_only=True)
class DistillationAgentFactory(AgentFactory["ActorCritic"]):
    num_steps_per_update: int = 24
    """Number of steps to collect before each update."""
    actor_hidden_dims: Sequence[int] = (256, 128)
    """Hidden dimensions of the actor network."""
    activation_fn: str | type[torch.nn.Module] = "ReLU"
    """Activation function to use in the actor network."""
    lr: float = 2e-4
    """Default Learning rate for the optimizer."""
    sampler_epochs: int = 1
    """Number of epochs to use for the sampler."""
    sampler_mini_batches: int = 8
    """Number of mini-batches to use for the sampler."""
    init_distribution_std: float | None = None
    """Standard deviation for initializing the action distribution."""
    expert_path: str = ""
    """Path to the expert model for distillation. Should be a JIT-exported model
    with an actor that takes in observations and outputs actions."""
    expert_observation_name: str = "observation"
    """The name of the observation key to use when feeding observations to the
    expert model during distillation."""
    normalize_observation: bool = False
    """Whether to normalize observations using running statistics."""
    max_grad_norm: float | None = 1.0
    """Maximum gradient norm for clipping."""

    def to_underlying(self) -> ActorCriticFactory:
        return ActorCriticFactory(
            num_steps_per_update=self.num_steps_per_update,
            actor_factory=cusrl.Actor.Factory(
                backbone_factory=cusrl.Mlp.Factory(
                    hidden_dims=self.actor_hidden_dims,
                    activation_fn=self.activation_fn,
                    ends_with_activation=True,
                ),
                distribution_factory=cusrl.NormalDist.Factory(init_std=self.init_distribution_std),
            ),
            critic_factory=cusrl.Value.Factory(
                backbone_factory=cusrl.StubModule.Factory(),
            ),
            optimizer_factory=AdamFactory(defaults={"lr": self.lr}),
            sampler=cusrl.AutoMiniBatchSampler(
                num_epochs=self.sampler_epochs,
                num_mini_batches=self.sampler_mini_batches,
            ),
            hooks=distillation_hook_suite(
                expert_path=self.expert_path,
                expert_observation_name=self.expert_observation_name,
                normalize_observation=self.normalize_observation,
                max_grad_norm=self.max_grad_norm,
            ),
            name=self.name,
            device=self.device,
            compile=self.compile,
            autocast=self.autocast,
        )

    def __call__(self, environment_spec: EnvironmentSpec) -> ActorCritic:
        return self.to_underlying()(environment_spec)
