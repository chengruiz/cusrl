from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

import cusrl
from cusrl.preset.optimizer import AdamFactory
from cusrl.template.actor_critic import ActorCritic, ActorCriticFactory
from cusrl.template.agent import AgentFactory as AgentFactoryBase
from cusrl.template.environment import EnvironmentSpec

__all__ = [
    "AgentFactory",
    "RecurrentAgentFactory",
    "hook_suite",
]


def hook_suite(
    orthogonal_init: bool = True,
    init_distribution_std: float | None = None,
    normalize_observation: bool = False,
    gae_gamma: float = 0.99,
    gae_lamda: float = 0.95,
    gae_lamda_value: float | None = None,
    normalize_advantage: bool = True,
    value_loss_weight: float = 0.5,
    value_loss_clip: float | None = None,
    surrogate_clip_ratio: float = 0.2,
    surrogate_loss_weight: float = 1.0,
    entropy_loss_weight: float = 0.01,
    max_grad_norm: float | None = 1.0,
    grad_clip_groups: dict[str, float] | None = None,
    desired_kl_divergence: float | None = None,
    empty_cuda_cache: bool = False,
) -> list[cusrl.template.Hook]:
    hooks = [
        cusrl.hook.ModuleInitialization(
            init_actor=orthogonal_init,
            init_critic=orthogonal_init,
            distribution_std=init_distribution_std,
        ),
        cusrl.hook.ObservationNormalization() if normalize_observation else None,
        cusrl.hook.ValueComputation(),
        cusrl.hook.GeneralizedAdvantageEstimation(
            gamma=gae_gamma,
            lamda=gae_lamda,
            lamda_value=gae_lamda_value,
        ),
        cusrl.hook.AdvantageNormalization() if normalize_advantage else None,
        cusrl.hook.ValueLoss(weight=value_loss_weight, loss_clip=value_loss_clip),
        cusrl.hook.OnPolicyPreparation(),
        cusrl.hook.PpoSurrogateLoss(clip_ratio=surrogate_clip_ratio, weight=surrogate_loss_weight),
        cusrl.hook.EntropyLoss(weight=entropy_loss_weight),
        cusrl.hook.GradientClipping(max_grad_norm, grad_clip_groups),
        cusrl.hook.OnPolicyStatistics(sampler=cusrl.AutoMiniBatchSampler()),
        cusrl.hook.AdaptiveLRSchedule(desired_kl_divergence) if desired_kl_divergence is not None else None,
        cusrl.hook.EmptyCudaCache() if empty_cuda_cache else None,
    ]
    return [hook for hook in hooks if hook is not None]


def get_distribution_factory(action_space_type: str):
    if action_space_type == "continuous":
        return cusrl.NormalDist.Factory()
    if action_space_type == "discrete":
        return cusrl.OneHotCategoricalDist.Factory()
    raise ValueError(f"Unsupported action space type '{action_space_type}'")


@dataclass(kw_only=True)
class AgentFactory(AgentFactoryBase[ActorCritic]):
    num_steps_per_update: int = 24
    """Number of environment steps to collect before performing an update."""
    actor_hidden_dims: Sequence[int] = (256, 128)
    """Hidden layer dimensions for the actor network."""
    critic_hidden_dims: Sequence[int] = (256, 128)
    """Hidden layer dimensions for the critic network."""
    activation_fn: str | type[torch.nn.Module] = "ReLU"
    """Activation function for the actor and critic networks."""
    action_space_type: str = "continuous"
    """Type of the action space, either 'continuous' or 'discrete'."""
    lr: float = 2e-4
    """Default learning rate for the optimizer."""
    sampler_epochs: int = 5
    """Number of epochs to perform on each update."""
    sampler_mini_batches: int = 4
    """Number of mini-batches to split the collected data into."""
    orthogonal_init: bool = True
    """Whether to use orthogonal initialization for the networks."""
    init_distribution_std: float | None = None
    """Standard deviation for initializing the action distribution."""
    normalize_observation: bool = False
    """Whether to normalize observations using running statistics."""
    gae_gamma: float = 0.99
    """Discount factor for Generalized Advantage Estimation (GAE)."""
    gae_lamda: float = 0.95
    """GAE lambda parameter for bias-variance trade-off."""
    gae_lamda_value: float | None = None
    """GAE lambda parameter for value function estimation. If None, uses the
    same value as gae_lamda."""
    normalize_advantage: bool = True
    """Whether to normalize advantages using running statistics."""
    value_loss_weight: float = 0.5
    """Weight for the value loss term in the total loss."""
    value_loss_clip: float | None = None
    """Clipping for value loss to limit the change in value estimates."""
    surrogate_clip_ratio: float = 0.2
    """Clipping ratio for the PPO surrogate loss."""
    surrogate_loss_weight: float = 1.0
    """Weight for the surrogate loss term in the total loss."""
    entropy_loss_weight: float = 0.01
    """Weight for the entropy loss term in the total loss."""
    max_grad_norm: float | None = 1.0
    """Maximum norm for gradient clipping. If None, no clipping is applied."""
    grad_clip_groups: dict[str, float] = field(default_factory=dict)
    """Dictionary specifying different max norm values for different parameter
    groups during gradient clipping."""
    desired_kl_divergence: float | None = None
    """Desired KL divergence between the old and new policy for adaptive
    learning rate adjustment. If None, no adaptive adjustment is performed."""

    def to_underlying(self) -> ActorCriticFactory:
        return ActorCriticFactory(
            num_steps_per_update=self.num_steps_per_update,
            actor_factory=cusrl.Actor.Factory(
                backbone_factory=cusrl.Mlp.Factory(
                    hidden_dims=self.actor_hidden_dims,
                    activation_fn=self.activation_fn,
                    ends_with_activation=True,
                ),
                distribution_factory=get_distribution_factory(self.action_space_type),
            ),
            critic_factory=cusrl.Value.Factory(
                backbone_factory=cusrl.Mlp.Factory(
                    hidden_dims=self.critic_hidden_dims,
                    activation_fn=self.activation_fn,
                    ends_with_activation=True,
                ),
            ),
            optimizer_factory=AdamFactory(defaults={"lr": self.lr}),
            sampler=cusrl.AutoMiniBatchSampler(
                num_epochs=self.sampler_epochs,
                num_mini_batches=self.sampler_mini_batches,
            ),
            hooks=hook_suite(
                orthogonal_init=self.orthogonal_init,
                init_distribution_std=self.init_distribution_std,
                normalize_observation=self.normalize_observation,
                gae_gamma=self.gae_gamma,
                gae_lamda=self.gae_lamda,
                gae_lamda_value=self.gae_lamda_value,
                normalize_advantage=self.normalize_advantage,
                value_loss_weight=self.value_loss_weight,
                value_loss_clip=self.value_loss_clip,
                surrogate_clip_ratio=self.surrogate_clip_ratio,
                surrogate_loss_weight=self.surrogate_loss_weight,
                entropy_loss_weight=self.entropy_loss_weight,
                max_grad_norm=self.max_grad_norm,
                grad_clip_groups=self.grad_clip_groups,
                desired_kl_divergence=self.desired_kl_divergence,
            ),
            name=self.name,
            device=self.device,
            compile=self.compile,
            autocast=self.autocast,
        )

    def __call__(self, environment_spec: EnvironmentSpec) -> ActorCritic:
        return self.to_underlying()(environment_spec)


@dataclass(kw_only=True)
class RecurrentAgentFactory(AgentFactoryBase[ActorCritic]):
    num_steps_per_update: int = 24
    """Number of environment steps to collect before performing an update."""
    rnn_type: str = "LSTM"
    """Type of RNN to use in the actor and critic, either 'LSTM' or 'GRU'."""
    actor_num_layers: int = 2
    """Number of RNN layers in the actor network."""
    actor_hidden_size: int = 256
    """Hidden size of the RNN layers in the actor network."""
    critic_num_layers: int = 2
    """Number of RNN layers in the critic network."""
    critic_hidden_size: int = 256
    """Hidden size of the RNN layers in the critic network."""
    action_space_type: str = "continuous"
    """Type of the action space, either 'continuous' or 'discrete'."""
    lr: float = 2e-4
    """Default learning rate for the optimizer."""
    sampler_epochs: int = 5
    """Number of epochs to perform on each update."""
    sampler_mini_batches: int = 4
    """Number of mini-batches to split the collected data into."""
    orthogonal_init: bool = True
    """Whether to use orthogonal initialization for the networks."""
    init_distribution_std: float | None = None
    """Standard deviation for initializing the action distribution."""
    normalize_observation: bool = False
    """Whether to normalize observations using running statistics."""
    gae_gamma: float = 0.99
    """Discount factor for Generalized Advantage Estimation (GAE)."""
    gae_lamda: float = 0.95
    """GAE lambda parameter for bias-variance trade-off."""
    gae_lamda_value: float | None = None
    """GAE lambda parameter for value function estimation. If None, uses the
    same value as gae_lamda."""
    normalize_advantage: bool = True
    """Whether to normalize advantages using running statistics."""
    value_loss_weight: float = 0.5
    """Weight for the value loss term in the total loss."""
    value_loss_clip: float | None = None
    """Clipping for value loss to limit the change in value estimates."""
    surrogate_clip_ratio: float = 0.2
    """Clipping ratio for the PPO surrogate loss."""
    entropy_loss_weight: float = 0.01
    """Weight for the entropy loss term in the total loss."""
    max_grad_norm: float | None = 1.0
    """Maximum norm for gradient clipping. If None, no clipping is applied."""
    grad_clip_groups: dict[str, float] = field(default_factory=dict)
    """Dictionary specifying different max norm values for different parameter
    groups during gradient clipping."""
    desired_kl_divergence: float | None = None
    """Desired KL divergence between the old and new policy for adaptive
    learning rate adjustment. If None, no adaptive adjustment is performed."""
    empty_cuda_cache: bool = True
    """Whether to empty CUDA cache after each update to reduce memory
    fragmentation."""

    def to_underlying(self) -> ActorCriticFactory:
        return ActorCriticFactory(
            num_steps_per_update=self.num_steps_per_update,
            actor_factory=cusrl.Actor.Factory(
                backbone_factory=cusrl.Rnn.Factory(
                    self.rnn_type,
                    num_layers=self.actor_num_layers,
                    hidden_size=self.actor_hidden_size,
                ),
                distribution_factory=get_distribution_factory(self.action_space_type),
            ),
            critic_factory=cusrl.Value.Factory(
                backbone_factory=cusrl.Rnn.Factory(
                    self.rnn_type,
                    num_layers=self.critic_num_layers,
                    hidden_size=self.critic_hidden_size,
                ),
            ),
            optimizer_factory=AdamFactory(defaults={"lr": self.lr}),
            sampler=cusrl.AutoMiniBatchSampler(
                num_epochs=self.sampler_epochs,
                num_mini_batches=self.sampler_mini_batches,
            ),
            hooks=hook_suite(
                orthogonal_init=self.orthogonal_init,
                init_distribution_std=self.init_distribution_std,
                normalize_observation=self.normalize_observation,
                gae_gamma=self.gae_gamma,
                gae_lamda=self.gae_lamda,
                gae_lamda_value=self.gae_lamda_value,
                normalize_advantage=self.normalize_advantage,
                value_loss_weight=self.value_loss_weight,
                value_loss_clip=self.value_loss_clip,
                surrogate_clip_ratio=self.surrogate_clip_ratio,
                entropy_loss_weight=self.entropy_loss_weight,
                max_grad_norm=self.max_grad_norm,
                grad_clip_groups=self.grad_clip_groups,
                desired_kl_divergence=self.desired_kl_divergence,
                empty_cuda_cache=self.empty_cuda_cache,
            ),
            name=self.name,
            device=self.device,
            compile=self.compile,
            autocast=self.autocast,
        )

    def __call__(self, environment_spec: EnvironmentSpec) -> ActorCritic:
        return self.to_underlying()(environment_spec)
