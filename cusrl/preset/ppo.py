from collections.abc import Iterable

import torch

import cusrl

__all__ = [
    "AgentFactory",
    "OptimizerFactory",
    "RecurrentAgentFactory",
    "hook_suite",
]


class OptimizerFactory(cusrl.template.OptimizerFactory):
    def __init__(self, lr: float = 2e-4):
        super().__init__(torch.optim.AdamW, defaults=dict(lr=lr))


def hook_suite(
    orthogonal_init: bool = True,
    init_distribution_std: float | None = None,
    normalize_observation: bool = False,
    gae_gamma: float = 0.99,
    gae_lamda: float = 0.95,
    gae_lamda_value: float | None = None,
    popart_alpha: float | None = None,
    normalize_advantage: bool = True,
    value_loss_weight: float = 0.5,
    value_loss_clip: float | None = None,
    surrogate_clip_ratio: float = 0.2,
    entropy_loss_weight: float = 0.01,
    max_grad_norm: float | None = 1.0,
    desired_kl_divergence: float | None = None,
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
            popart_alpha=popart_alpha,
        ),
        cusrl.hook.AdvantageNormalization() if normalize_advantage else None,
        cusrl.hook.ValueLoss(weight=value_loss_weight, loss_clip=value_loss_clip),
        cusrl.hook.OnPolicyPreparation(),
        cusrl.hook.PpoSurrogateLoss(clip_ratio=surrogate_clip_ratio),
        cusrl.hook.EntropyLoss(weight=entropy_loss_weight),
        cusrl.hook.GradientClipping(max_grad_norm) if max_grad_norm is not None else None,
        cusrl.hook.OnPolicyStatistics(sampler=cusrl.AutoMiniBatchSampler()),
        cusrl.hook.AdaptiveLRSchedule(desired_kl_divergence) if desired_kl_divergence is not None else None,
    ]
    return [hook for hook in hooks if hook is not None]


def get_distribution_factory(action_space_type: str):
    if action_space_type == "continuous":
        return cusrl.NormalDist.Factory()
    elif action_space_type == "discrete":
        return cusrl.OneHotCategoricalDist.Factory()
    else:
        raise ValueError(f"Unsupported action space type '{action_space_type}'.")


class AgentFactory(cusrl.template.ActorCritic.Factory):
    def __init__(
        self,
        num_steps_per_update: int = 24,
        actor_hidden_dims: Iterable[int] = (256, 128),
        critic_hidden_dims: Iterable[int] = (256, 128),
        activation_fn: str | type[torch.nn.Module] = "ReLU",
        action_space_type: str = "continuous",
        lr: float = 2e-4,
        sampler_epochs: int = 3,
        sampler_mini_batches: int = 8,
        orthogonal_init: bool = True,
        init_distribution_std: float | None = None,
        normalize_observation: bool = False,
        gae_gamma: float = 0.99,
        gae_lamda: float = 0.95,
        gae_lamda_value: float | None = None,
        popart_alpha: float | None = None,
        normalize_advantage: bool = True,
        value_loss_weight: float = 0.5,
        value_loss_clip: float | None = None,
        surrogate_clip_ratio: float = 0.2,
        entropy_loss_weight: float = 0.01,
        max_grad_norm: float | None = 1.0,
        desired_kl_divergence: float | None = None,
        device: str | torch.device | None = None,
        compile: bool = False,
        autocast: bool | torch.dtype = False,
    ):
        super().__init__(
            num_steps_per_update=num_steps_per_update,
            actor_factory=cusrl.Actor.Factory(
                backbone_factory=cusrl.Mlp.Factory(
                    hidden_dims=actor_hidden_dims,
                    activation_fn=activation_fn,
                    ends_with_activation=True,
                ),
                distribution_factory=get_distribution_factory(action_space_type),
            ),
            critic_factory=cusrl.Value.Factory(
                backbone_factory=cusrl.Mlp.Factory(
                    hidden_dims=critic_hidden_dims,
                    activation_fn=activation_fn,
                    ends_with_activation=True,
                ),
            ),
            optimizer_factory=OptimizerFactory(lr=lr),
            sampler=cusrl.AutoMiniBatchSampler(
                num_epochs=sampler_epochs,
                num_mini_batches=sampler_mini_batches,
            ),
            hooks=hook_suite(
                orthogonal_init=orthogonal_init,
                init_distribution_std=init_distribution_std,
                normalize_observation=normalize_observation,
                gae_gamma=gae_gamma,
                gae_lamda=gae_lamda,
                gae_lamda_value=gae_lamda_value,
                popart_alpha=popart_alpha,
                normalize_advantage=normalize_advantage,
                value_loss_weight=value_loss_weight,
                value_loss_clip=value_loss_clip,
                surrogate_clip_ratio=surrogate_clip_ratio,
                entropy_loss_weight=entropy_loss_weight,
                max_grad_norm=max_grad_norm,
                desired_kl_divergence=desired_kl_divergence,
            ),
            device=device,
            compile=compile,
            autocast=autocast,
        )


class RecurrentAgentFactory(cusrl.template.ActorCritic.Factory):
    def __init__(
        self,
        num_steps_per_update: int = 24,
        rnn_type: str = "LSTM",
        actor_num_layers: int = 2,
        actor_hidden_size: int = 256,
        critic_num_layers: int = 2,
        critic_hidden_size: int = 256,
        action_space_type: str = "continuous",
        lr: float = 2e-4,
        sampler_epochs: int = 3,
        sampler_mini_batches: int = 8,
        orthogonal_init: bool = True,
        init_distribution_std: float | None = None,
        normalize_observation: bool = False,
        gae_gamma: float = 0.99,
        gae_lamda: float = 0.95,
        gae_lamda_value: float | None = None,
        popart_alpha: float | None = None,
        normalize_advantage: bool = True,
        value_loss_weight: float = 0.5,
        value_loss_clip: float | None = None,
        surrogate_clip_ratio: float = 0.2,
        entropy_loss_weight: float = 0.01,
        max_grad_norm: float | None = 1.0,
        desired_kl_divergence: float | None = None,
        device: str | torch.device | None = None,
        compile: bool = False,
        autocast: bool | torch.dtype = False,
    ):
        super().__init__(
            num_steps_per_update=num_steps_per_update,
            actor_factory=cusrl.Actor.Factory(
                backbone_factory=cusrl.Rnn.Factory(
                    rnn_type,
                    num_layers=actor_num_layers,
                    hidden_size=actor_hidden_size,
                ),
                distribution_factory=get_distribution_factory(action_space_type),
            ),
            critic_factory=cusrl.Value.Factory(
                backbone_factory=cusrl.Rnn.Factory(
                    rnn_type,
                    num_layers=critic_num_layers,
                    hidden_size=critic_hidden_size,
                ),
            ),
            optimizer_factory=OptimizerFactory(lr=lr),
            sampler=cusrl.AutoMiniBatchSampler(
                num_epochs=sampler_epochs,
                num_mini_batches=sampler_mini_batches,
            ),
            hooks=hook_suite(
                orthogonal_init=orthogonal_init,
                init_distribution_std=init_distribution_std,
                normalize_observation=normalize_observation,
                gae_gamma=gae_gamma,
                gae_lamda=gae_lamda,
                gae_lamda_value=gae_lamda_value,
                popart_alpha=popart_alpha,
                normalize_advantage=normalize_advantage,
                value_loss_weight=value_loss_weight,
                value_loss_clip=value_loss_clip,
                surrogate_clip_ratio=surrogate_clip_ratio,
                entropy_loss_weight=entropy_loss_weight,
                max_grad_norm=max_grad_norm,
                desired_kl_divergence=desired_kl_divergence,
            ),
            device=device,
            compile=compile,
            autocast=autocast,
        )
