from dataclasses import dataclass

from torch import Tensor

from cusrl.module.distribution import Distribution, DistributionFactoryLike
from cusrl.module.module import Module, ModuleFactory, ModuleFactoryLike
from cusrl.utils.typing import Memory, Slice

__all__ = ["Actor"]


@dataclass(slots=True)
class ActorFactory(ModuleFactory["Actor"]):
    backbone_factory: ModuleFactoryLike
    distribution_factory: DistributionFactoryLike
    latent_dim: int | None = None

    def __call__(self, input_dim: int | None, output_dim: int) -> "Actor":
        backbone = self.backbone_factory(input_dim, self.latent_dim)
        distribution = self.distribution_factory(backbone.output_dim, output_dim)
        return Actor(backbone, distribution)


class Actor(Module):
    Factory = ActorFactory
    backbone: Module
    distribution: Distribution

    def __init__(
        self,
        backbone: Module,
        distribution: Distribution,
    ):
        super().__init__(
            backbone.input_dim,
            distribution.output_dim,
            backbone.is_recurrent,
        )
        self.backbone = backbone.rnn_compatible()
        self.distribution = distribution
        self.latent_dim = self.backbone.output_dim

    def to_distributed(self):
        if not self.is_distributed:
            self.is_distributed = True
            self.backbone = self.backbone.to_distributed()
            self.distribution = self.distribution.to_distributed()
        return self

    def clear_intermediate_repr(self):
        super().clear_intermediate_repr()
        self.backbone.clear_intermediate_repr()
        self.distribution.clear_intermediate_repr()

    def forward(
        self,
        *args,
        forward_type: str | None = "forward",
        **kwargs,
    ):
        if forward_type == "forward":
            return self._forward_impl(*args, **kwargs)
        if forward_type == "explore":
            return self._explore_impl(*args, **kwargs)
        if forward_type == "act":
            return self._act_impl(*args, **kwargs)
        if forward_type == "act_deterministic":
            return self._act_impl(*args, **kwargs, deterministic=True)
        raise ValueError(f"Unknown forward type: {forward_type}")

    def explore(
        self,
        observation: Tensor,
        memory: Memory = None,
        deterministic: bool = False,
        backbone_kwargs: dict | None = None,
        distribution_kwargs: dict | None = None,
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor], Memory]:
        return self(
            observation,
            memory=memory,
            deterministic=deterministic,
            backbone_kwargs=backbone_kwargs,
            distribution_kwargs=distribution_kwargs,
            forward_type="explore",
        )

    def act(
        self,
        observation: Tensor,
        memory: Memory = None,
        deterministic: bool = False,
        backbone_kwargs: dict | None = None,
        distribution_kwargs: dict | None = None,
    ) -> tuple[Tensor, Memory]:
        return self(
            observation,
            memory=memory,
            deterministic=deterministic,
            backbone_kwargs=backbone_kwargs,
            distribution_kwargs=distribution_kwargs,
            forward_type="act",
        )

    def _forward_impl(
        self,
        observation: Tensor,
        memory: Memory = None,
        done: Tensor | None = None,
        backbone_kwargs: dict | None = None,
        distribution_kwargs: dict | None = None,
    ) -> tuple[tuple[Tensor, Tensor], Memory]:
        latent, memory = self.backbone(
            observation,
            memory=memory,
            done=done,
            **(backbone_kwargs or {}),
        )

        action_mean, action_std = self.distribution(
            latent,
            observation=observation,
            **(distribution_kwargs or {}),
        )

        self.intermediate_repr["backbone.output"] = latent
        self.intermediate_repr["backbone.intermediate_repr"] = self.backbone.intermediate_repr
        self.intermediate_repr["distribution.intermediate_repr"] = self.distribution.intermediate_repr
        return (action_mean, action_std), memory

    def _explore_impl(
        self,
        observation: Tensor,
        memory: Memory = None,
        deterministic: bool = False,
        backbone_kwargs: dict | None = None,
        distribution_kwargs: dict | None = None,
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor], Memory]:
        latent, memory = self.backbone(
            observation,
            memory=memory,
            **(backbone_kwargs or {}),
        )
        if deterministic:
            action_mean, action_std = self.distribution(
                latent,
                observation=observation,
                **(distribution_kwargs or {}),
            )
            action = self.distribution.determine(
                latent,
                observation=observation,
                **(distribution_kwargs or {}),
            )
            logp = self.distribution.calc_logp(action_mean, action_std, action)
        else:
            action_mean, action_std, action, logp = self.distribution.sample(
                latent,
                observation=observation,
                **(distribution_kwargs or {}),
            )

        self.intermediate_repr["backbone.output"] = latent
        self.intermediate_repr["backbone.intermediate_repr"] = self.backbone.intermediate_repr
        self.intermediate_repr["distribution.intermediate_repr"] = self.distribution.intermediate_repr
        return (action_mean, action_std), (action, logp), memory

    def _act_impl(
        self,
        observation: Tensor,
        memory: Memory = None,
        deterministic: bool = False,
        backbone_kwargs: dict | None = None,
        distribution_kwargs: dict | None = None,
    ) -> tuple[Tensor, Memory]:
        _, (action, _), memory = self._explore_impl(
            observation,
            memory=memory,
            deterministic=deterministic,
            backbone_kwargs=backbone_kwargs,
            distribution_kwargs=distribution_kwargs,
        )
        return action, memory

    def calc_logp(self, action_mean, action_std, action):
        return self.distribution.calc_logp(action_mean, action_std, action)

    def calc_entropy(self, action_mean, action_std):
        return self.distribution.calc_entropy(action_mean, action_std)

    def step_memory(self, observation, memory=None, **kwargs):
        return self.backbone.step_memory(observation, memory, **kwargs)

    def reset_memory(self, memory: Memory, done: Slice | Tensor | None = None):
        self.backbone.reset_memory(memory, done)

    def set_distribution_std(self, action_std):
        if hasattr(self.distribution, "set_std"):
            self.distribution.set_std(action_std)

    def clamp_distribution_std(self, lb=None, ub=None, indices=slice(None)):
        if hasattr(self.distribution, "clamp_std"):
            self.distribution.clamp_std(lb, ub, indices)
