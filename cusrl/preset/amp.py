from collections.abc import Callable, Iterable
from dataclasses import dataclass

import cusrl
from cusrl.preset.ppo import PpoAgentFactory
from cusrl.template.actor_critic import ActorCriticFactory
from cusrl.utils.typing import Array, Slice

__all__ = ["AmpAgentFactory"]


@dataclass(kw_only=True)
class AmpAgentFactory(PpoAgentFactory):
    extrinsic_reward_scale: float = 1.0
    """Scale of the extrinsic reward when combined with the intrinsic reward
    from Adversarial Motion Prior (AMP)."""
    amp_discriminator_hidden_dims: Iterable[int] = (256, 128)
    """Hidden dimensions of the MLP discriminator used in AMP."""
    amp_dataset_source: str | Array | Callable[[], Array] | None = None
    """Source of the dataset for AMP."""
    amp_state_indices: Slice | None = None
    """Indices of the state to be used as input to the AMP discriminator."""
    amp_batch_size: int = 512
    """Batch size for training the AMP discriminator."""
    amp_reward_scale: float = 1.0
    """Scale of the intrinsic reward when combined with the extrinsic reward."""
    amp_loss_weight: float = 1.0
    """Weight of the AMP loss when combined with the PPO loss."""
    amp_grad_penalty_weight: float = 5.0
    """Weight of the gradient penalty term in the AMP discriminator loss."""

    def to_underlying(self) -> ActorCriticFactory:
        underlying = super().to_underlying()
        underlying.register_hook(
            cusrl.hook.RewardShaping(scale=self.extrinsic_reward_scale),
            before="value_computation",
        )
        underlying.register_hook(
            cusrl.hook.AdversarialMotionPrior(
                discriminator_factory=cusrl.Mlp.Factory(
                    hidden_dims=self.amp_discriminator_hidden_dims,
                    activation_fn=self.activation_fn,
                ),
                dataset_source=self.amp_dataset_source,
                state_indices=self.amp_state_indices,
                batch_size=self.amp_batch_size,
                reward_scale=self.amp_reward_scale,
                loss_weight=self.amp_loss_weight,
                grad_penalty_weight=self.amp_grad_penalty_weight,
            ),
            after="reward_shaping",
        )
        return underlying
