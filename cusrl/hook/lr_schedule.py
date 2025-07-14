import math

import torch

from cusrl.template import ActorCritic, Hook
from cusrl.utils import distributed

__all__ = ["AdaptiveLRSchedule", "MiniBatchWiseLRSchedule", "ThresholdLRSchedule"]


class KLDivergenceBasedLRSchedule(Hook[ActorCritic]):
    MUTABLE_ATTRS = ["desired_kl_divergence"]

    def __init__(
        self,
        desired_kl_divergence: float = 0.01,
        scale_all_params: bool = False,
    ):
        if desired_kl_divergence <= 0:
            raise ValueError("'desired_kl_divergence' must be positive.")

        self.desired_kl_divergence = desired_kl_divergence
        self.scale_all_params = scale_all_params
        self._lr_scale = 1.0

    def post_update(self):
        kl_divergence = self.agent.metrics["kl_divergence"].mean.clone()
        distributed.reduce_mean_(kl_divergence)
        scale = self._compute_scale(kl_divergence.item())
        self._scale_lr_of_parameters(scale)
        self.agent.record(lr_scale=self._lr_scale)

    def _compute_scale(self, kl_divergence: float) -> float | None:
        raise NotImplementedError

    def _scale_lr_of_parameters(self, scale: float | None):
        if scale is None or scale == 1.0:
            return

        self._lr_scale *= scale
        for param_group in self.agent.optimizer.param_groups:
            if self.scale_all_params or any(name.startswith("actor.") for name in param_group["param_names"]):
                param_group["lr"] *= scale

    def state_dict(self):
        return {"lr_scale": self._lr_scale}

    def load_state_dict(self, state_dict):
        self._lr_scale = state_dict["lr_scale"]


class ThresholdLRSchedule(KLDivergenceBasedLRSchedule):
    """Adjusts the learning rate based on thresholded KL divergence.

    Args:
        desired_kl_divergence (float, optional):
            Target KL divergence to maintain. Defaults to 0.01.
        threshold (float, optional):
            Ratio threshold (>1) for deciding when to adjust. Defaults to 1.2.
        scale_factor (float, optional):
            Multiplicative factor (>1) for scaling the LR. Defaults to 1.1.
        scale_all_params (bool, optional):
            If True, scales all optimizer parameter groups; otherwise only
            scales actor parameter groups. Defaults to False.
    """

    MUTABLE_ATTRS = ["desired_kl_divergence"]

    def __init__(
        self,
        desired_kl_divergence: float = 0.01,
        threshold: float = 1.2,
        scale_factor: float = 1.1,
        scale_all_params: bool = False,
    ):
        super().__init__(desired_kl_divergence, scale_all_params)
        if threshold <= 1:
            raise ValueError("'threshold' must be greater than 1.")
        if scale_factor <= 1:
            raise ValueError("'scale_factor' must be greater than 1.")

        self.threshold = threshold
        self.scale_factor = scale_factor

    def _compute_scale(self, kl_divergence: float):
        if kl_divergence > self.desired_kl_divergence * self.threshold:
            return 1 / self.scale_factor
        if kl_divergence < self.desired_kl_divergence / self.threshold:
            return self.scale_factor
        return None


class AdaptiveLRSchedule(KLDivergenceBasedLRSchedule):
    """Adaptively adjusts the learning rate based on accumulated KL divergence error.

    Args:
        desired_kl_divergence (float, optional):
            Target KL divergence to maintain. Defaults to 0.01.
        threshold (float, optional):
            Positive threshold for accumulated log-error before scaling. Defaults to 1.0.
        scale_factor (float, optional):
            Positive coefficient controlling adjustment magnitude. Defaults to 0.2.
        scale_all_params (bool, optional):
            If True, scales all optimizer parameter groups; otherwise only scales actor
            parameter groups. Defaults to False.
    """

    MUTABLE_ATTRS = ["desired_kl_divergence"]

    def __init__(
        self,
        desired_kl_divergence: float = 0.01,
        threshold: float = 1.0,
        scale_factor: float = 0.2,
        scale_all_params: bool = False,
    ):
        super().__init__(desired_kl_divergence, scale_all_params)
        if threshold <= 0:
            raise ValueError("'threshold' must be positive.")
        if scale_factor <= 0:
            raise ValueError("'scale_factor' must be positive.")

        self.threshold = threshold
        self.scale_factor = scale_factor
        self.accumulated_log_error = 0.0
        self.count = 0

    def _compute_scale(self, kl_divergence: float):
        self.accumulated_log_error += math.log(kl_divergence / self.desired_kl_divergence)
        self.count += 1
        if self.threshold > self.accumulated_log_error > -self.threshold:
            return None
        average_log_error = self.accumulated_log_error / self.count
        scale = math.exp(-min(max(average_log_error, -1.0), 1.0) * self.scale_factor)
        self.accumulated_log_error = 0.0
        self.count = 0
        return scale


class MiniBatchWiseLRSchedule(ThresholdLRSchedule):
    """Applies a threshold-based LR schedule on a per-mini-batch KL divergence.
    Modified from (RSL-RL)[https://github.com/leggedrobotics/rsl_rl].

    Args:
        desired_kl_divergence (float, optional):
            Target KL divergence per mini-batch. Defaults to 0.01.
        threshold (float, optional):
            Ratio threshold for deciding scaling per batch. Defaults to 2.0.
        scale_factor (float, optional):
            Multiplicative factor for scaling the LR. Defaults to 1.5.
    """

    def __init__(
        self,
        desired_kl_divergence: float = 0.01,
        threshold: float = 2.0,
        scale_factor: float = 1.5,
    ):
        super().__init__(
            desired_kl_divergence,
            threshold=threshold,
            scale_factor=scale_factor,
            scale_all_params=True,
        )

    def post_init(self):
        self.agent.hook["OnPolicyPreparation"].calculate_kl_divergence = True

    def post_update(self):
        pass

    def objective(self, batch):
        with torch.inference_mode():
            kl_divergence = batch["kl_divergence"].mean()
        distributed.reduce_mean_(kl_divergence)
        scale = self._compute_scale(kl_divergence.item())
        self._scale_lr_of_parameters(scale)
        self.agent.record(lr_scale=self._lr_scale)
