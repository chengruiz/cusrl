from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeAlias, TypedDict, TypeVar

import torch
from torch import Tensor, distributions, nn
from torch.nn.functional import one_hot

from cusrl.nn.layer.bijector import Bijector, make_bijector
from cusrl.nn.module.module import Module, ModuleFactory

__all__ = [
    "AdaptiveNormalDist",
    "Distribution",
    "DistributionFactoryLike",
    "NormalDist",
    "OneHotCategoricalDist",
]

DistributionT = TypeVar("DistributionT", bound="Distribution")
DistributionParamsT = TypeVar("DistributionParamsT")


class DistributionFactory(ModuleFactory[DistributionT]):
    def __call__(self, input_dim: int | None = None, output_dim: int | None = None):
        raise NotImplementedError


DistributionFactoryLike: TypeAlias = Callable[[int | None, int | None], "Distribution"]


class Distribution(Module, Generic[DistributionParamsT]):
    """Abstract base class for probability distributions.

    Args:
        input_dim (int):
            The dimensionality of the input space.
        output_dim (int):
            The dimensionality of the output action space.
    """

    Factory = DistributionFactory

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(input_dim, output_dim)
        self.mean_head = nn.Linear(input_dim, output_dim)

    def forward(self, backbone_feat: Tensor, **kwargs) -> DistributionParamsT:
        """Computes the parameters of the distribution from backbone features.

        This method must be implemented by subclasses. It should return the
        parameters that define the distribution (e.g., mean and standard
        deviation).

        Args:
            backbone_feat (Tensor):
                The input tensor of backbone features.
            **kwargs:
                Additional keyword arguments.

        Returns:
            dist_params (DistributionParamsT):
                A dictionary containing the distribution parameters.
        """
        raise NotImplementedError

    def sample(self, backbone_feat: Tensor, **kwargs) -> tuple[DistributionParamsT, tuple[Tensor, Tensor]]:
        """Computes distribution parameters and samples from the distribution.

        Args:
            backbone_feat (Tensor):
                The input tensor of backbone features.
            **kwargs:
                Additional keyword arguments.

        Returns:
            dist_params (DistributionParamsT):
                The parameters of the distribution produced by forward().
            sampled (tuple[Tensor, Tensor]):
                A tuple containing the sampled action and its log-probability.
        """
        dist_params = self(backbone_feat, **kwargs)
        action, logp = self.sample_from_dist(dist_params)
        return dist_params, (action, logp)

    def sample_from_dist(self, dist_params: DistributionParamsT) -> tuple[Tensor, Tensor]:
        """Samples an action and computes its log-probability from the
        distribution defined by dist_params.

        Args:
            dist_params (DistributionParamsT):
                The parameters of the distribution as produced by forward().

        Outputs:
            - **action** (Tensor):
                A sample from the distribution.
            - **logp** (Tensor):
                The log-probability of the sample under the distribution, a
                tensor of shape :math:`(..., 1)`.
        """
        raise NotImplementedError

    def compute_logp(self, dist_params: DistributionParamsT, sample: Tensor) -> Tensor:
        """Computes the log probability of a sample given the distribution
        parameters.

        Args:
            dist_params (DistributionParamsT):
                The parameters of the distribution.
            sample (Tensor):
                The sample for which to compute the log probability.

        Returns:
            log_probability (Tensor):
                The log probability of the sample  under the distribution, a
                tensor of shape :math:`(..., 1)`.
        """
        raise NotImplementedError

    def compute_entropy(self, dist_params: DistributionParamsT) -> Tensor:
        r"""Computes the entropy of the distribution. Defaults to a single-
        sample Monte Carlo estimate.

        .. math::
            H(P) = -\int P(x) \log P(x) dx

        Args:
            dist_params (DistributionParamsT):
                The parameters of the distribution.

        Returns:
            entropy (Tensor):
                The entropy of the distribution, a tensor of shape
                :math:`(..., 1)`.
        """
        _, logp = self.sample_from_dist(dist_params)
        return -logp

    def compute_kl_div(self, dist_params1: DistributionParamsT, dist_params2: DistributionParamsT) -> Tensor:
        r"""Computes the KL divergence between two distributions P and Q.
        Defaults to a single-sample Monte Carlo estimate.

        .. math::
            D_{KL}(P || Q) = \int P(x) \log \frac{P(x)}{Q(x)} dx

        Args:
            dist_params1 (DistributionParamsT):
                The parameters of the first distribution (P).
            dist_params2 (DistributionParamsT):
                The parameters of the second distribution (Q).

        Returns:
            kl_divergence (Tensor):
                The KL divergence between the two distributions, a tensor of
                shape :math:`(..., 1)`.
        """
        sample, logp = self.sample_from_dist(dist_params1)
        logq = self.compute_logp(dist_params2, sample)
        return logp - logq

    def determine(self, backbone_feat: Tensor, **kwargs) -> Tensor:
        """Returns the deterministic action for given backbone features.

        Args:
            backbone_feat (Tensor):
                The input tensor of backbone features.
            **kwargs:
                Additional keyword arguments.

        Returns:
            action (Tensor):
                The action with the highest probability.
        """
        return self.mean_head(backbone_feat)

    def deterministic(self):
        return DeterministicWrapper(self)


class DeterministicWrapper(nn.Module):
    def __init__(self, distribution: Distribution):
        super().__init__()
        self.dist = distribution

    def forward(self, backbone_feat, **kwargs):
        return self.dist.determine(backbone_feat, **kwargs)


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
    def __init__(
        self,
        output_dim: int,
        init_std: float | None = None,
        bijector: str | Bijector | None = None,
    ):
        super().__init__()
        self.bijector = make_bijector(bijector)
        self.param = nn.Parameter(torch.ones(output_dim) * self.bijector.inverse(init_std or 1.0))

    def forward(self, input: Tensor):
        return self.bijector(self.param.repeat(*input.shape[:-1], 1))

    def __repr__(self):
        return f"StddevVector(bijector={self.bijector})"


@dataclass(slots=True)
class NormalDistFactory(DistributionFactory["NormalDist"]):
    init_std: float | None = None
    bijector: str | Bijector | None = None

    def __call__(self, input_dim: int | None = None, output_dim: int | None = None):
        assert input_dim is not None and output_dim is not None
        return NormalDist(input_dim, output_dim, init_std=self.init_std, bijector=self.bijector)


class NormalDist(_Normal):
    Factory = NormalDistFactory

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        init_std: float | None = None,
        bijector: str | Bijector | None = None,
    ):
        super().__init__(input_dim, output_dim)
        self.std: StddevVector = StddevVector(output_dim, init_std=init_std, bijector=bijector)

    def forward(self, backbone_feat: Tensor, **kwargs):
        return MeanStdDict(mean=self.mean_head(backbone_feat), std=self.std(backbone_feat))


@dataclass(slots=True)
class AdaptiveNormalDistFactory(DistributionFactory["AdaptiveNormalDist"]):
    bijector: str | Bijector | None = "exp"
    backward: bool = True

    def __call__(self, input_dim: int | None = None, output_dim: int | None = None):
        assert input_dim is not None and output_dim is not None
        return AdaptiveNormalDist(input_dim, output_dim, bijector=self.bijector, backward=self.backward)


class AdaptiveNormalDist(_Normal):
    Factory = AdaptiveNormalDistFactory

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        init_std: float | None = None,
        bijector: str | Bijector | None = "exp",
        backward: bool = True,
    ):
        super().__init__(input_dim, output_dim)

        self.std_head = nn.Linear(input_dim, output_dim)
        self.bijector = make_bijector(bijector)
        self.backward = backward

        self.std_head.weight.data.zero_()
        self.std_head.bias.data[:] = self.bijector.inverse(init_std or 1.0)

    def forward(self, backbone_feat: Tensor, **kwargs):
        action_mean = self.mean_head(backbone_feat)
        if not self.backward:
            backbone_feat = backbone_feat.detach()
        std = self.std_head(backbone_feat)
        return MeanStdDict(mean=action_mean, std=self.bijector(std))


class OneHotCategoricalDistFactory(DistributionFactory["OneHotCategoricalDist"]):
    def __call__(self, input_dim: int | None = None, output_dim: int | None = None):
        assert input_dim is not None and output_dim is not None
        return OneHotCategoricalDist(input_dim, output_dim)


class LogitsDict(TypedDict):
    logits: Tensor


class OneHotCategoricalDist(Distribution[LogitsDict]):
    Factory = OneHotCategoricalDistFactory

    @classmethod
    def _dist(cls, dist_params) -> distributions.OneHotCategorical:
        return distributions.OneHotCategorical(logits=dist_params["logits"], validate_args=False)

    def forward(self, backbone_feat: Tensor, **kwargs):
        logits: Tensor = self.mean_head(backbone_feat)
        return LogitsDict(logits=logits)

    def determine(self, backbone_feat: Tensor, **kwargs) -> Tensor:
        logits: Tensor = self.mean_head(backbone_feat)
        mode = one_hot(logits.argmax(dim=-1), logits.size(-1))
        return mode

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
