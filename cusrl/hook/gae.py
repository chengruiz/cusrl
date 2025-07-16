from typing import Any

import torch

from cusrl.template import ActorCritic, Hook
from cusrl.utils import ExponentialMovingNormalizer

__all__ = ["GeneralizedAdvantageEstimation"]


@torch.jit.script
def _generalized_advantage_estimation(
    reward: torch.Tensor,
    done: torch.Tensor,
    value: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    lamda: float,
) -> torch.Tensor:
    not_done = done.logical_not()
    advantage = reward + next_value * gamma - value
    for step in range(advantage.size(0) - 2, -1, -1):
        advantage[step] += not_done[step] * (gamma * lamda) * advantage[step + 1]
    return advantage


class GeneralizedAdvantageEstimation(Hook[ActorCritic]):
    """A hook that computes advantages and returns using Generalized Advantage Estimation (GAE).

    GAE is described in:
    "High-Dimensional Continuous Control Using Generalized Advantage Estimation",
    https://arxiv.org/abs/1506.02438

    Distinct lambda values can be enabled to individually control the bias-variance trade-offs
    for policy and value function, described in:
    "DNA: Proximal Policy Optimization with a Dual Network Architecture"
    https://proceedings.neurips.cc/paper_files/paper/2022/hash/e95475f5fb8edb9075bf9e25670d4013-Abstract-Conference.html

    PopArt normalization can be applied to the value function, described in:
    "Learning values across many orders of magnitude",
    https://proceedings.neurips.cc/paper/2016/hash/5227b6aaf294f5f027273aebf16015f2-Abstract.html

    Args:
        gamma (float, optional):
            Discount factor for future rewards, in [0, 1). Defaults to 0.99.
        lamda (float, optional):
            Smoothing factor for advantage estimation, in [0, 1]. Defaults to 0.95.
        lamda_value (float | None, optional):
            Smoothing factor for value function calculation, in [0, 1].
            If None, the same value as `lamda` is used. Defaults to None.
        recompute (bool, optional):
            If True, recompute advantages and returns after each update. Defaults to False.
        popart_alpha (float | None, optional):
            If not None, applies PopArt normalization to the value function with the specified
            alpha. Defaults to None, which means no normalization is applied.
    """

    MODULES = ["value_rms"]
    MUTABLE_ATTRS = ["gamma", "lamda"]

    def __init__(
        self,
        gamma: float = 0.99,
        lamda: float = 0.95,
        lamda_value: float | None = None,
        recompute: bool = False,
        popart_alpha: float | None = None,
    ):
        if gamma < 0 or gamma >= 1:
            raise ValueError(f"Invalid gamma value {gamma}, which should be in [0, 1).")
        if lamda < 0 or lamda > 1:
            raise ValueError(f"Invalid lambda value {lamda}, which should be in [0, 1].")
        if lamda_value is not None and (lamda_value < 0 or lamda_value > 1):
            raise ValueError(f"Invalid lambda value for value function {lamda_value}, which should be in [0, 1].")

        self.gamma = gamma
        self.lamda = lamda
        self.lamda_value = lamda_value
        self.recompute = recompute
        self.popart_alpha = popart_alpha
        self.value_rms: ExponentialMovingNormalizer | None = None

    def init(self):
        if self.popart_alpha is not None:
            self.value_rms = self.__make_normalizer(self.agent.value_dim)
            self.agent.critic.value_rms = self.__make_normalizer(self.agent.value_dim)

    def __make_normalizer(self, num_channels: int):
        return ExponentialMovingNormalizer(num_channels, alpha=self.popart_alpha).to(self.agent.device)

    def pre_update(self, buffer):
        if not self.recompute:
            self._compute_advantage_and_return(buffer)

    def objective(self, batch: dict[str, Any]):
        if self.recompute:
            self._compute_advantage_and_return(batch)

    def post_update(self):
        if self.value_rms is not None:
            old_value_rms: ExponentialMovingNormalizer = self.agent.critic.value_rms
            old_mean, old_std = old_value_rms.mean, old_value_rms.std
            # Adjust value head weights and biases
            new_mean, new_std = self.value_rms.mean, self.value_rms.std
            value_head = self.agent.critic.value_head
            with torch.no_grad():
                value_head.weight.data.mul_(old_std / new_std)
                value_head.bias.data.mul_(old_std).add_(old_mean).sub_(new_mean).div_(new_std)
            old_value_rms.load_state_dict(self.value_rms.state_dict())

    @torch.no_grad()
    def _compute_advantage_and_return(self, data):
        value = data["value"]
        next_value = data["next_value"]
        if (value_rms := self.agent.critic.value_rms) is not None:
            value = value_rms.unnormalize(value)
            next_value = value_rms.unnormalize(next_value)

        data["advantage"] = _generalized_advantage_estimation(
            reward=data["reward"],
            done=data["done"],
            value=value,
            next_value=next_value,
            gamma=self.gamma,
            lamda=self.lamda,
        )

        data["return"] = value + (
            data["advantage"]
            if self.lamda_value is None
            else _generalized_advantage_estimation(
                reward=data["reward"],
                done=data["done"],
                value=value,
                next_value=next_value,
                gamma=self.gamma,
                lamda=self.lamda_value,
            )
        )

        if value_rms is not None:
            self.value_rms.update(data["return"])
            value_rms.normalize_(data["return"])
