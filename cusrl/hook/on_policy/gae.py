import torch

from cusrl.template import ActorCritic, Hook

__all__ = ["GeneralizedAdvantageEstimation"]


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
    """Computes advantages and returns using Generalized Advantage Estimation.

    Generalized Advantage Estimation (GAE) is described in:
    "High-Dimensional Continuous Control Using Generalized Advantage
    Estimation",
    https://arxiv.org/abs/1506.02438

    Distinct lambda values can be enabled to individually control the bias-
    variance trade-offs for policy and value function, described in:
    "DNA: Proximal Policy Optimization with a Dual Network Architecture"
    https://proceedings.neurips.cc/paper_files/paper/2022/hash/e95475f5fb8edb9075bf9e25670d4013-Abstract-Conference.html

    Args:
        gamma (float, optional):
            Discount factor for future rewards, in :math:`[0, 1)`. Defaults to
            ``0.99``.
        lamda (float, optional):
            Smoothing factor for advantage estimation, in :math:`[0, 1]`.
            Defaults to ``0.95``.
        lamda_value (float | None, optional):
            Smoothing factor for value function calculation, in :math:`[0, 1]`.
            If ``None``, the same value as ``lamda`` is used. Defaults to
            ``None``.
        recompute (bool, optional):
            If ``True``, recompute advantages and returns after each update.
            Defaults to ``False``.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        lamda: float = 0.95,
        lamda_value: float | None = None,
        recompute: bool = False,
    ):
        if gamma < 0 or gamma >= 1:
            raise ValueError(f"'gamma' must be in [0, 1); got {gamma}")
        if lamda < 0 or lamda > 1:
            raise ValueError(f"'lamda' must be in [0, 1]; got {lamda}")
        if lamda_value is not None and (lamda_value < 0 or lamda_value > 1):
            raise ValueError(f"'lamda_value' must be in [0, 1]; got {lamda_value}")

        super().__init__(training_only=True)
        self.recompute = recompute

        # Mutable attributes
        self.gamma: float = gamma
        self.lamda: float = lamda
        self.lamda_value: float | None = lamda_value
        self.register_mutable("gamma")
        self.register_mutable("lamda")
        self.register_mutable("lamda_value")

    def pre_update(self, buffer):
        if not self.recompute:
            self._compute_advantage_and_return(buffer)

    def objective(self, metadata, batch):
        if self.recompute:
            self._compute_advantage_and_return(batch)

    @torch.no_grad()
    def _compute_advantage_and_return(self, data):
        value = data["value"]
        next_value = data["next_value"]

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
