from typing import Any, cast

import torch
from torch import nn

from cusrl.template import ActorCritic, Buffer, Hook
from cusrl.utils.helper import get_or
from cusrl.utils.nest import map_nested
from cusrl.utils.typing import Memory

__all__ = ["ValueComputation", "ValueLoss"]


class ValueComputation(Hook[ActorCritic]):
    """Computes and stores state-value estimates using the critic.

    This hook handles the value computation for terminal and truncated states.

    Args:
        termination_value (float):
            The value to assign to terminal states.
        bootstrap_truncated_states (bool):
            If `True`, bootstraps the value of truncated states using the critic.
    """

    def __init__(
        self,
        *,
        termination_value: float = 0.0,
        bootstrap_truncated_states: bool = True,
    ):
        self.termination_value = termination_value
        self.bootstrap_truncated_states = bootstrap_truncated_states
        self._critic_memory: Memory = None

    def init(self):
        if self.agent.environment_spec.final_state_is_missing:
            self.bootstrap_truncated_states = False

    def post_act(self, transition):
        critic = self.agent.critic
        with self.agent.autocast():
            value, next_critic_memory = critic(
                get_or(transition, "state", "observation"),
                memory=self._critic_memory,
            )

        transition["value"] = value
        transition["critic_memory"] = self._critic_memory
        transition["next_critic_memory"] = next_critic_memory
        self._critic_memory = next_critic_memory

    def post_step(self, transition):
        self.agent.critic.reset_memory(self._critic_memory, transition["done"])

    @torch.no_grad()
    def pre_update(self, buffer: Buffer):
        critic = self.agent.critic
        value = cast(torch.Tensor, buffer["value"])
        if (next_value := buffer.get("next_value")) is None:
            buffer["next_value"] = torch.zeros_like(cast(torch.Tensor, value))
            next_value = buffer["next_value"]
        next_value = cast(torch.Tensor, next_value)
        next_state = cast(torch.Tensor, get_or(buffer, "next_state", "next_observation"))
        terminated = cast(torch.Tensor, buffer["terminated"]).squeeze(-1)
        truncated = cast(torch.Tensor, buffer["truncated"]).squeeze(-1)

        next_value[:-1] = value[1:]
        with self.agent.autocast():
            next_value[-1] = critic.evaluate(next_state[-1], memory=self._critic_memory)
        termination_value = value.new_full([value.size(-1)], self.termination_value)
        if critic.value_rms is not None:
            critic.value_rms.normalize_(termination_value)
        next_value[terminated] = termination_value
        if self.bootstrap_truncated_states:
            next_memory = buffer.get("next_critic_memory")
            if next_memory is not None:
                # fmt: off
                next_memory = map_nested(
                    next_memory,
                    lambda mem: mem        # [ N, ..., B, C]
                        .unsqueeze(1)      # [ N, 1, ..., B, C]
                        .transpose(1, -2)  # [ N, B, ..., 1, C]
                        [truncated]        # [ M, ..., 1, C]
                        .transpose(0, -2)  # [ 1, ..., M, C]
                        .squeeze(0)        # [ ..., M, C]
                        .contiguous(),
                )
                # fmt: on

            with self.agent.autocast():
                next_value[truncated] = critic.evaluate(next_state[truncated], memory=next_memory)
        else:
            next_value[truncated] = value[truncated]


def _clipped_value_loss(
    value: torch.Tensor,
    curr_value: torch.Tensor,
    return_: torch.Tensor,
    loss_clip: float,
):
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

    MUTABLE_ATTRS = ["weight", "value_loss_clip"]

    def __init__(self, weight: float = 0.5, loss_clip: float | None = None):
        self.weight = weight
        self.loss_clip = loss_clip

    def objective(self, batch: dict[str, Any]):
        critic = self.agent.critic
        with self.agent.autocast():
            curr_value = batch.get("curr_value")
            if curr_value is None:
                curr_value = critic.evaluate(
                    batch.get("state", batch["observation"]),
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
            self.agent.record(
                value=curr_value[..., 0],
                **{f"value_{i}": curr_value[..., i] for i in range(1, curr_value.size(-1))},
                value_loss=value_loss,
            )

        return value_loss
