from typing import cast

import torch
from torch import Tensor, nn

from cusrl.template import ActorCritic, Buffer, Hook
from cusrl.utils.dict_utils import get_first
from cusrl.utils.nest import map_nested
from cusrl.utils.recurrent import apply_sequence_batch_mask, set_sequence_batch_masked_
from cusrl.utils.typing import Memory

__all__ = ["ValueComputation", "ValueLoss"]


class ValueComputation(Hook[ActorCritic]):
    """Computes and stores state-value estimates using the critic.

    This hook handles the value computation for terminal and truncated states.

    Args:
        termination_value (float):
            The value to assign to terminal states.
        bootstrap_truncated_states (bool):
            If `True`, bootstraps the value of truncated states using the
            critic.
    """

    def __init__(
        self,
        *,
        termination_value: float = 0.0,
        bootstrap_truncated_states: bool = True,
    ):
        super().__init__()
        self.termination_value = termination_value
        self.bootstrap_truncated_states = bootstrap_truncated_states
        self._critic_memory: Memory = None

    def init(self):
        if self.agent.environment_spec.final_state_is_missing:
            self.bootstrap_truncated_states = False

    def post_act(self, transition):
        critic = self.agent.critic
        state = cast(Tensor, get_first(transition, "state", "observation"))
        with self.agent.autocast():
            value, next_critic_memory = critic(state.unsqueeze(0), memory=self._critic_memory)

        transition["value"] = value.squeeze(0)
        transition["critic_memory"] = self._critic_memory
        transition["next_critic_memory"] = next_critic_memory
        self._critic_memory = next_critic_memory

    def post_step(self, transition):
        self.agent.critic.reset_memory(self._critic_memory, transition["done"])

    @torch.no_grad()
    def pre_update(self, buffer: Buffer):
        critic = self.agent.critic
        value = cast(Tensor, buffer["value"])
        if (next_value := buffer.get("next_value")) is None:
            buffer["next_value"] = torch.zeros_like(cast(Tensor, value))
            next_value = buffer["next_value"]
        next_value = cast(Tensor, next_value)
        next_state = cast(Tensor, get_first(buffer, "next_state", "next_observation"))
        terminated = cast(Tensor, buffer["terminated"]).squeeze(-1)
        truncated = cast(Tensor, buffer["truncated"]).squeeze(-1)

        next_value[:-1] = value[1:]
        with self.agent.autocast():
            next_value[-1] = critic.evaluate(next_state[-1], memory=self._critic_memory)
        termination_value = value.new_full([value.size(-1)], self.termination_value)
        if critic.value_rms is not None:
            critic.value_rms.normalize_(termination_value)
        set_sequence_batch_masked_(next_value, terminated, termination_value)
        if truncated.any():
            if self.bootstrap_truncated_states:
                if (next_memory := buffer.get("next_critic_memory")) is not None:
                    next_memory = map_nested(
                        lambda memory: apply_sequence_batch_mask(memory, truncated).contiguous(),
                        next_memory,
                    )

                truncated_next_state = apply_sequence_batch_mask(next_state, truncated)
                with self.agent.autocast():
                    truncated_next_value = critic.evaluate(truncated_next_state, memory=next_memory)
                set_sequence_batch_masked_(next_value, truncated, truncated_next_value)
            else:
                set_sequence_batch_masked_(next_value, truncated, apply_sequence_batch_mask(value, truncated))


def _clipped_value_loss(value: Tensor, curr_value: Tensor, return_: Tensor, loss_clip: float):
    clipped_value = value + (curr_value - value).clamp(-loss_clip, loss_clip)
    value_loss = (curr_value - return_).square()
    value_loss_clipped = (clipped_value - return_).square()
    return torch.max(value_loss, value_loss_clipped).mean()


class ValueLoss(Hook[ActorCritic]):
    """Calculates the value function loss for an actor-critic agent.

    This hook supports two common methods for loss calculation: standard Mean
    Squared Error (MSE) and a clipped value loss, which is often used in
    algorithms like Proximal Policy Optimization (PPO) to stabilize training by
    limiting the update size.

    Args:
        weight (float, optional):
            The weight applied to the value loss. Defaults to 0.5.
        loss_clip (float | None, optional):
            If specified, uses a clipped value loss instead of standard MSE.
            Defaults to None, which means standard MSE is used.
    """

    def __init__(self, weight: float = 0.5, loss_clip: float | None = None):
        if weight <= 0:
            raise ValueError("'weight' must be positive.")
        if loss_clip is not None and loss_clip <= 0:
            raise ValueError("'loss_clip' must be positive or None.")
        super().__init__()

        # Mutable attributes
        self.weight: float
        self.loss_clip: float | None
        self.register_mutable("weight", weight)
        self.register_mutable("loss_clip", loss_clip)

    def objective(self, batch):
        critic = self.agent.critic
        with self.agent.autocast():
            curr_value = critic.evaluate(
                get_first(batch, "state", "observation"),
                memory=batch.get("critic_memory"),
                done=batch["done"],
            )
            batch["curr_value"] = curr_value

            value_loss = (
                nn.functional.mse_loss(batch["return"], curr_value)
                if self.loss_clip is None
                else _clipped_value_loss(batch["value"], curr_value, batch["return"], self.loss_clip)
            ) * self.weight

        with torch.no_grad():
            if critic.value_rms is not None:
                curr_value = critic.value_rms.unnormalize(curr_value)
            self.agent.record(value=curr_value, value_loss=value_loss)
            if (value_dim := curr_value.size(-1)) != 1:
                self.agent.record(**{f"value.{i}": curr_value[..., i] for i in range(value_dim)})

        return value_loss
