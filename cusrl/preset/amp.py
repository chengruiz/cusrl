from collections.abc import Callable, Iterable
from dataclasses import dataclass

import cusrl
from cusrl.preset import ppo
from cusrl.utils.typing import Array, Slice

__all__ = ["AgentFactory"]


@dataclass
class AgentFactory(ppo.AgentFactory):
    extrinsic_reward_scale: float = 1.0
    amp_discriminator_hidden_dims: Iterable[int] = (256, 128)
    amp_dataset_source: str | Array | Callable[[], Array] | None = None
    amp_state_indices: Slice | None = None
    amp_batch_size: int = 512
    amp_reward_scale: float = 1.0
    amp_loss_weight: float = 1.0
    amp_grad_penalty_weight: float = 5.0

    def __post_init__(self):
        super().__post_init__()
        self.register_hook(
            cusrl.hook.RewardShaping(scale=self.extrinsic_reward_scale),
            before="value_computation",
        )
        self.register_hook(
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
