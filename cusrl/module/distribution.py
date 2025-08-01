from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeAlias, TypedDict, TypeVar

import torch
from torch import Tensor, distributions, nn
from torch.nn.functional import one_hot

from cusrl import utils
from cusrl.module.bijector import Bijector, get_bijector
from cusrl.module.module import Module, ModuleFactory
from cusrl.utils.typing import NestedTensor

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
        raise NotImplementedError

    @classmethod
    def compute_logp(cls, dist_params: ParamType, sample: torch.Tensor) -> Tensor:
        """Computes the log probability of a sample given the distribution
        parameters.

        Args:
            dist_params (ParamType):
                The parameters of the distribution.
            sample (Tensor):
                The sample for which to compute the log probability.

        Returns:
            log_probability (Tensor):
                The log probability of the sample.
        """
        raise NotImplementedError

    @classmethod
    def compute_entropy(cls, dist_params: ParamType) -> Tensor:
        """Computes the entropy of the distribution.

        Args:
            dist_params (ParamType):
                The parameters of the distribution.

        Returns:
            entropy (Tensor):
                The entropy of the distribution.
        """
        raise NotImplementedError

    @classmethod
    def compute_kl_div(cls, dist_params1: ParamType, dist_params2: ParamType) -> Tensor:
        r"""Computes the KL divergence between two distributions P and Q.

        .. math::
            D_{KL}(P || Q) = \int P(x) \log \frac{P(x)}{Q(x)} dx

        Args:
            dist_params1 (Tensor):
                The parameters of the first distribution (P).
            dist_params2 (Tensor):
                The parameters of the second distribution (Q).

        Returns:
            kl_divergence (Tensor):
                The KL divergence between the two distributions.
        """
        raise NotImplementedError

    def determine(self, latent: Tensor, **kwargs) -> Tensor:
        """Returns the deterministic action for a given latent state.

        Args:
            latent (Tensor):
                The input tensor from the latent space.
            **kwargs:
                Additional keyword arguments.

        Returns:
            action (Tensor): The action with the highest probability (mean of
                the distribution).
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

    @classmethod
    def compute_logp(cls, dist_params, sample) -> Tensor:
        return cls._dist(dist_params).log_prob(sample).sum(dim=-1, keepdim=True)

    @classmethod
    def compute_entropy(cls, dist_params) -> Tensor:
        return cls._dist(dist_params).entropy().sum(dim=-1, keepdim=True)

    @classmethod
    def compute_kl_div(cls, dist_params1, dist_params2) -> Tensor:
        kl = distributions.kl_divergence(cls._dist(dist_params1), cls._dist(dist_params2))
        return kl.sum(dim=-1, keepdim=True)


class StddevVector(nn.Module):
    def __init__(self, output_dim: int, bijector: str | Bijector | None = "exp"):
        super().__init__()
        self.bijector = get_bijector(bijector)
        self.param = nn.Parameter(torch.ones(output_dim) * self.bijector.inverse(1.0))

    def forward(self, input: Tensor):
        return self.bijector(self.param.repeat(*input.shape[:-1], 1))

    def clamp(self, lb: float | None = None, ub: float | None = None, indices=slice(None)):
        if lb is None and ub is None:
            return
        if lb is not None:
            lb = self.bijector.inverse(lb)
        if ub is not None:
            ub = self.bijector.inverse(ub)
        self.param.data[indices].clamp_(min=lb, max=ub)

    def set(self, value):
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
    std: StddevVector

    def __init__(self, input_dim: int, output_dim: int, bijector: str | Bijector | None = "exp"):
        super().__init__(input_dim, output_dim)
        self.std = StddevVector(output_dim, bijector=bijector)

    def forward(self, latent: Tensor, **kwargs) -> NestedTensor:
        return {"mean": self.mean_head(latent), "std": self.std(latent)}

    def to_distributed(self):
        if not self.is_distributed:
            super().to_distributed()
            self.std = utils.make_distributed(self.std)
        return self

    def set_std(self, std):
        self.std.set(std)

    def clamp_std(self, lb: float | None = None, ub: float | None = None, indices=slice(None)):
        self.std.clamp(lb=lb, ub=ub, indices=indices)


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
    std_head: nn.Linear

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bijector: str | Bijector | None = "exp",
        backward: bool = True,
    ):
        super().__init__(input_dim, output_dim)

        self.std_head = nn.Linear(input_dim, output_dim)
        self.bijector = get_bijector(bijector)
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

    def forward(self, latent: Tensor, **kwargs) -> NestedTensor:
        action_mean = self.mean_head(latent)
        if not self.backward:
            latent = latent.detach()
        std = self.std_head(latent)
        return {"mean": action_mean, "std": self.bijector(std)}

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
        return {"logit": logit}

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

    @classmethod
    def compute_logp(cls, dist_params, sample: Tensor) -> Tensor:
        logp = cls._dist(dist_params).log_prob(sample).unsqueeze(-1)
        return logp

    @classmethod
    def compute_entropy(cls, dist_params) -> Tensor:
        return cls._dist(dist_params).entropy().unsqueeze(-1)

    @classmethod
    def compute_kl_div(cls, dist_params1, dist_params2) -> Tensor:
        return distributions.kl_divergence(cls._dist(dist_params1), cls._dist(dist_params2)).unsqueeze(-1)
