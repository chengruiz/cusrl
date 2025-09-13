from collections.abc import Callable, Iterable

import torch

import cusrl
from cusrl.preset import ppo
from cusrl.utils.typing import Array, Slice

__all__ = ["AgentFactory"]


class AgentFactory(ppo.AgentFactory):
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
        extrinsic_reward_scale: float = 1.0,
        amp_discriminator_hidden_dims: Iterable[int] = (256, 128),
        amp_dataset_source: str | Array | Callable[[], Array] | None = None,
        amp_state_indices: Slice | None = None,
        amp_batch_size: int = 512,
        amp_reward_scale: float = 1.0,
        amp_loss_weight: float = 1.0,
        amp_grad_penalty_weight: float = 5.0,
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
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation_fn=activation_fn,
            action_space_type=action_space_type,
            lr=lr,
            sampler_epochs=sampler_epochs,
            sampler_mini_batches=sampler_mini_batches,
            orthogonal_init=orthogonal_init,
            init_distribution_std=init_distribution_std,
            normalize_observation=normalize_observation,
            popart_alpha=popart_alpha,
            gae_gamma=gae_gamma,
            gae_lamda=gae_lamda,
            gae_lamda_value=gae_lamda_value,
            normalize_advantage=normalize_advantage,
            value_loss_weight=value_loss_weight,
            value_loss_clip=value_loss_clip,
            surrogate_clip_ratio=surrogate_clip_ratio,
            entropy_loss_weight=entropy_loss_weight,
            max_grad_norm=max_grad_norm,
            desired_kl_divergence=desired_kl_divergence,
            device=device,
            compile=compile,
            autocast=autocast,
        )
        self.register_hook(
            cusrl.hook.RewardShaping(scale=extrinsic_reward_scale),
            before="value_computation",
        )
        self.register_hook(
            cusrl.hook.AdversarialMotionPrior(
                discriminator_factory=cusrl.Mlp.Factory(
                    hidden_dims=amp_discriminator_hidden_dims,
                    activation_fn=activation_fn,
                ),
                dataset_source=amp_dataset_source,
                state_indices=amp_state_indices,
                batch_size=amp_batch_size,
                reward_scale=amp_reward_scale,
                loss_weight=amp_loss_weight,
                grad_penalty_weight=amp_grad_penalty_weight,
            ),
            after="reward_shaping",
        )
        self._kwargs.update({
            "amp_discriminator_hidden_dims": amp_discriminator_hidden_dims,
            "amp_dataset_source": amp_dataset_source,
            "amp_state_indices": amp_state_indices,
            "amp_batch_size": amp_batch_size,
            "amp_reward_scale": amp_reward_scale,
            "amp_loss_weight": amp_loss_weight,
            "amp_grad_penalty_weight": amp_grad_penalty_weight,
        })
