from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias, TypeVar

import torch
from torch import Tensor, distributions, nn
from torch.nn.functional import one_hot

from cusrl import utils
from cusrl.module.bijector import Bijector, get_bijector
from cusrl.module.module import Module, ModuleFactory

__all__ = [
    "AdaptiveNormalDist",
    "Distribution",
    "DistributionFactoryLike",
    "NormalDist",
    "OneHotCategoricalDist",
]

DistributionType = TypeVar("DistributionType", bound="Distribution")


class DistributionFactory(ModuleFactory[DistributionType]):
    def __call__(self, input_dim: int, output_dim: int) -> DistributionType:
        raise NotImplementedError


class Distribution(Module):
    Factory = DistributionFactory

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(input_dim, output_dim)
        self.mean_head = nn.Linear(input_dim, output_dim)

    def forward(self, latent, **kwargs) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def sample(self, latent, **kwargs) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        action_mean, action_std = self(latent, **kwargs)
        action, logp = self.sample_from_dist(action_mean, action_std)
        return action_mean, action_std, action, logp

    def sample_from_dist(self, mean, std) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    @classmethod
    def calc_logp(cls, mean, std, sample) -> Tensor:
        raise NotImplementedError

    @classmethod
    def calc_entropy(cls, mean, std) -> Tensor:
        raise NotImplementedError

    @classmethod
    def calc_kl_div(cls, mean1, std1, mean2, std2) -> Tensor:
        r"""
        .. math::
            D_{KL}(P || Q) = \int P(x) \log \frac{P(x)}{Q(x)} dx
        """
        raise NotImplementedError

    def determine(self, latent: Tensor, **kwargs) -> Tensor:
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


class _Normal(Distribution):
    @classmethod
    def _dist(cls, mean: Tensor, std: Tensor) -> distributions.Normal:
        return distributions.Normal(mean, std, validate_args=False)

    def sample_from_dist(self, mean, std) -> tuple[Tensor, Tensor]:
        dist = self._dist(mean, std)
        sample = dist.rsample()
        logp = dist.log_prob(sample).sum(dim=-1, keepdim=True)
        return sample, logp

    @classmethod
    def calc_logp(cls, mean, std, sample) -> Tensor:
        return cls._dist(mean, std).log_prob(sample).sum(dim=-1, keepdim=True)

    @classmethod
    def calc_entropy(cls, mean, std) -> Tensor:
        return cls._dist(mean, std).entropy().sum(dim=-1, keepdim=True)

    @classmethod
    def calc_kl_div(cls, mean1, std1, mean2, std2) -> Tensor:
        kl = distributions.kl_divergence(cls._dist(mean1, std1), cls._dist(mean2, std2))
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

    def forward(self, latent, **kwargs) -> tuple[Tensor, Tensor]:
        return self.mean_head(latent), self.std(latent)

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

    def forward(self, latent, **kwargs) -> tuple[Tensor, Tensor]:
        action_mean = self.mean_head(latent)
        if not self.backward:
            latent = latent.detach()
        std = self.std_head(latent)
        return action_mean, self.bijector(std)

    def set_std(self, std):
        self.std_head.weight.data.zero_()
        self.std_head.bias.data[:] = self.bijector.inverse(std)


class OneHotCategoricalDistFactory(DistributionFactory["OneHotCategoricalDist"]):
    def __call__(self, input_dim: int, output_dim: int):
        return OneHotCategoricalDist(input_dim, output_dim)


class OneHotCategoricalDist(Distribution):
    Factory = OneHotCategoricalDistFactory

    def forward(self, latent: Tensor, **kwargs) -> tuple[Tensor, Tensor]:
        logit: Tensor = self.mean_head(latent)
        mode = one_hot(logit.argmax(dim=-1), logit.size(-1))
        return mode, logit

    def determine(self, latent: Tensor, **kwargs) -> Tensor:
        logit: Tensor = self.mean_head(latent)
        mode = one_hot(logit.argmax(dim=-1), logit.size(-1))
        return mode

    @classmethod
    def _dist(cls, logit: Tensor) -> distributions.OneHotCategorical:
        return distributions.OneHotCategorical(logits=logit, validate_args=False)

    def sample_from_dist(self, mode: Tensor, logit: Tensor) -> tuple[Tensor, Tensor]:
        dist = self._dist(logit)
        action = dist.sample()
        logp = dist.log_prob(action).unsqueeze(-1)
        return action, logp

    @classmethod
    def calc_logp(cls, mode: Tensor, logit: Tensor, sample: Tensor) -> Tensor:
        logp = cls._dist(logit).log_prob(sample).unsqueeze(-1)
        return logp

    @classmethod
    def calc_entropy(cls, mode: Tensor, logit: Tensor) -> Tensor:
        return cls._dist(logit).entropy().unsqueeze(-1)

    @classmethod
    def calc_kl_div(cls, mode1: Tensor, logit1: Tensor, mode2: Tensor, logit2: Tensor) -> Tensor:
        return distributions.kl_divergence(cls._dist(logit1), cls._dist(logit2)).unsqueeze(-1)
