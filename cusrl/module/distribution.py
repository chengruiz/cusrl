from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeAlias, TypedDict, TypeVar

import torch
from torch import Tensor, distributions, nn
from torch.nn.functional import one_hot

from cusrl import utils
from cusrl.module.bijector import Bijector, make_bijector
from cusrl.module.module import Module, ModuleFactory

__all__ = [
    "AdaptiveNormalDist",
    "Distribution",
    "DistributionFactoryLike",
    "NormalDist",
    "OneHotCategoricalDist",
]

DistributionType = TypeVar("DistributionType", bound="Distribution")
ParamType = TypeVar("ParamType")


class DistributionFactory(ModuleFactory[DistributionType]):
    def __call__(self, input_dim: int, output_dim: int) -> DistributionType:
        raise NotImplementedError


class Distribution(Module, Generic[ParamType]):
    """Abstract base class for probability distributions.

    Args:
        input_dim (int):
            The dimensionality of the input latent space.
        output_dim (int):
            The dimensionality of the output action space.
    """

    Factory = DistributionFactory

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(input_dim, output_dim)
        self.mean_head = nn.Linear(input_dim, output_dim)

    def forward(self, latent: Tensor, **kwargs) -> ParamType:
        """Computes the parameters of the distribution from a latent tensor.

        This method must be implemented by subclasses. It should return the
        parameters that define the distribution (e.g., mean and standard
        deviation).

        Args:
            latent (Tensor):
                The input tensor from the latent space.
            **kwargs:
                Additional keyword arguments.

        Returns:
            dist_params (ParamType):
                A dictionary containing the distribution parameters.
        """
        raise NotImplementedError

    def sample(self, latent: Tensor, **kwargs) -> tuple[ParamType, tuple[Tensor, Tensor]]:
        dist_params = self(latent, **kwargs)
        action, logp = self.sample_from_dist(dist_params)
        return dist_params, (action, logp)

    def sample_from_dist(self, dist_params: ParamType) -> tuple[Tensor, Tensor]:
        """Samples an action and computes its log-probability from the
        distribution defined by dist_params.

        Args:
            dist_params (ParamType):
                The parameters of the distribution as produced by forward().

        Returns:
            action (Tensor):
                A sample from the distribution with shape [..., output_dim].
            logp (Tensor):
                The log-probability of the sample under the distribution, with
                shape [..., 1].
        """
        raise NotImplementedError

    def compute_logp(self, dist_params: ParamType, sample: Tensor) -> Tensor:
        """Computes the log probability of a sample given the distribution
        parameters.

        Args:
            dist_params (ParamType):
                The parameters of the distribution.
            sample (Tensor):
                The sample for which to compute the log probability.

        Returns:
            log_probability (Tensor):
                The log probability of the sample, a tensor of shape [..., 1].
        """
        raise NotImplementedError

    def compute_entropy(self, dist_params: ParamType) -> Tensor:
        r"""Computes the entropy of the distribution. Defaults to a single-
        sample Monte Carlo estimate.

        .. math::
            H(P) = -\int P(x) \log P(x) dx

        Args:
            dist_params (ParamType):
                The parameters of the distribution.

        Returns:
            entropy (Tensor):
                The entropy of the distribution, a tensor of shape [..., 1].
        """
        _, logp = self.sample_from_dist(dist_params)
        return -logp

    def compute_kl_div(self, dist_params1: ParamType, dist_params2: ParamType) -> Tensor:
        r"""Computes the KL divergence between two distributions P and Q.
        Defaults to a single-sample Monte Carlo estimate.

        .. math::
            D_{KL}(P || Q) = \int P(x) \log \frac{P(x)}{Q(x)} dx

        Args:
            dist_params1 (ParamType):
                The parameters of the first distribution (P).
            dist_params2 (ParamType):
                The parameters of the second distribution (Q).

        Returns:
            kl_divergence (Tensor):
                The KL divergence between the two distributions, a tensor of
                shape [..., 1].
        """
        sample, logp = self.sample_from_dist(dist_params1)
        logq = self.compute_logp(dist_params2, sample)
        return logp - logq

    def determine(self, latent: Tensor, **kwargs) -> Tensor:
        """Returns the deterministic action for a given latent state.

        Args:
            latent (Tensor):
                The input tensor from the latent space.
            **kwargs:
                Additional keyword arguments.

        Returns:
            action (Tensor):
                The action with the highest probability.
        """
        return self.mean_head(latent)

    def deterministic(self):
        return DeterministicWrapper(self)

    def to_distributed(self):
        if not self.is_distributed:
            self.is_distributed = True
            self.mean_head = utils.make_distributed(self.mean_head)
        return self


DistributionFactoryLike: TypeAlias = Callable[[int, int], Distribution]


class DeterministicWrapper(nn.Module):
    def __init__(self, distribution: Distribution):
        super().__init__()
        self.dist = distribution

    def forward(self, latent, **kwargs):
        return self.dist.determine(latent, **kwargs)


class MeanStdDict(TypedDict):
    mean: Tensor
    std: Tensor


class _Normal(Distribution[MeanStdDict]):
    @classmethod
    def _dist(cls, dist_params) -> distributions.Normal:
        return distributions.Normal(dist_params["mean"], dist_params["std"], validate_args=False)

    def sample_from_dist(self, dist_params) -> tuple[Tensor, Tensor]:
        dist = self._dist(dist_params)
        sample = dist.rsample()
        logp = dist.log_prob(sample).sum(dim=-1, keepdim=True)
        return sample, logp

    def compute_logp(self, dist_params, sample) -> Tensor:
        return self._dist(dist_params).log_prob(sample).sum(dim=-1, keepdim=True)

    def compute_entropy(self, dist_params) -> Tensor:
        return self._dist(dist_params).entropy().sum(dim=-1, keepdim=True)

    def compute_kl_div(self, dist_params1, dist_params2) -> Tensor:
        kl = distributions.kl_divergence(self._dist(dist_params1), self._dist(dist_params2))
        return kl.sum(dim=-1, keepdim=True)


class StddevVector(nn.Module):
    def __init__(self, output_dim: int, bijector: str | Bijector | None = "exp"):
        super().__init__()
        self.bijector = make_bijector(bijector)
        self.param = nn.Parameter(torch.ones(output_dim) * self.bijector.inverse(1.0))

    def forward(self, input: Tensor):
        return self.bijector(self.param.repeat(*input.shape[:-1], 1))

    def clamp_(self, lb: float | None = None, ub: float | None = None, indices=slice(None)):
        if lb is None and ub is None:
            return
        if lb is not None:
            lb = self.bijector.inverse(lb)
        if ub is not None:
            ub = self.bijector.inverse(ub)
        self.param.data[indices].clamp_(min=lb, max=ub)

    def fill_(self, value):
        self.param.data[:] = self.bijector.inverse(value)

    def __repr__(self):
        return f"StddevVector(bijector={self.bijector})"


@dataclass(slots=True)
class NormalDistFactory(DistributionFactory["NormalDist"]):
    bijector: str | Bijector | None = "exp"

    def __call__(self, input_dim: int, output_dim: int):
        return NormalDist(input_dim, output_dim, bijector=self.bijector)


class NormalDist(_Normal):
    Factory = NormalDistFactory

    def __init__(self, input_dim: int, output_dim: int, bijector: str | Bijector | None = "exp"):
        super().__init__(input_dim, output_dim)
        self.std: StddevVector = StddevVector(output_dim, bijector=bijector)

    def forward(self, latent: Tensor, **kwargs):
        return MeanStdDict(mean=self.mean_head(latent), std=self.std(latent))

    def to_distributed(self):
        if not self.is_distributed:
            super().to_distributed()
            self.std = utils.make_distributed(self.std)
        return self

    def set_std(self, std):
        self.std.fill_(std)

    def clamp_std(self, lb: float | None = None, ub: float | None = None, indices=slice(None)):
        self.std.clamp_(lb=lb, ub=ub, indices=indices)


@dataclass(slots=True)
class AdaptiveNormalDistFactory(DistributionFactory["AdaptiveNormalDist"]):
    bijector: str | Bijector | None = "exp"
    backward: bool = True

    def __call__(self, input_dim: int, output_dim: int):
        return AdaptiveNormalDist(
            input_dim,
            output_dim,
            bijector=self.bijector,
            backward=self.backward,
        )


class AdaptiveNormalDist(_Normal):
    Factory = AdaptiveNormalDistFactory

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bijector: str | Bijector | None = "exp",
        backward: bool = True,
    ):
        super().__init__(input_dim, output_dim)

        self.std_head: nn.Linear = nn.Linear(input_dim, output_dim)
        self.bijector = make_bijector(bijector)
        self.backward = backward

    def to_distributed(self):
        if not self.is_distributed:
            super().to_distributed()
            self.std_head = utils.make_distributed(self.std_head)
        return self

    def clear_intermediate_repr(self):
        super().clear_intermediate_repr()
        if isinstance(self.std_head, Module):
            self.std_head.clear_intermediate_repr()

    def forward(self, latent: Tensor, **kwargs):
        action_mean = self.mean_head(latent)
        if not self.backward:
            latent = latent.detach()
        std = self.std_head(latent)
        return MeanStdDict(mean=action_mean, std=self.bijector(std))

    def set_std(self, std):
        self.std_head.weight.data.zero_()
        self.std_head.bias.data[:] = self.bijector.inverse(std)


class OneHotCategoricalDistFactory(DistributionFactory["OneHotCategoricalDist"]):
    def __call__(self, input_dim: int, output_dim: int):
        return OneHotCategoricalDist(input_dim, output_dim)


class LogitDict(TypedDict):
    logit: Tensor


class OneHotCategoricalDist(Distribution[LogitDict]):
    Factory = OneHotCategoricalDistFactory

    def forward(self, latent: Tensor, **kwargs):
        logit: Tensor = self.mean_head(latent)
        return LogitDict(logit=logit)

    def determine(self, latent: Tensor, **kwargs) -> Tensor:
        logit: Tensor = self.mean_head(latent)
        mode = one_hot(logit.argmax(dim=-1), logit.size(-1))
        return mode

    @classmethod
    def _dist(cls, dist_params) -> distributions.OneHotCategorical:
        return distributions.OneHotCategorical(logits=dist_params["logit"], validate_args=False)

    def sample_from_dist(self, dist_params) -> tuple[Tensor, Tensor]:
        dist = self._dist(dist_params)
        action = dist.sample()
        logp = dist.log_prob(action).unsqueeze(-1)
        return action, logp

    def compute_logp(self, dist_params, sample: Tensor) -> Tensor:
        logp = self._dist(dist_params).log_prob(sample).unsqueeze(-1)
        return logp

    def compute_entropy(self, dist_params) -> Tensor:
        return self._dist(dist_params).entropy().unsqueeze(-1)

    def compute_kl_div(self, dist_params1, dist_params2) -> Tensor:
        return distributions.kl_divergence(self._dist(dist_params1), self._dist(dist_params2)).unsqueeze(-1)
