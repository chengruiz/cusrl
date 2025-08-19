from dataclasses import dataclass

from torch import Tensor

from cusrl.module.distribution import Distribution, DistributionFactoryLike
from cusrl.module.module import Module, ModuleFactory, ModuleFactoryLike
from cusrl.utils.dict_utils import prefix_dict_keys
from cusrl.utils.typing import Memory, NestedTensor, Slice

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
    """An actor model for reinforcement learning.

    The Actor class encapsulates a policy network, which maps observations to
    actions. It is composed of a `backbone` module that processes observations
    to produce a latent representation, and a `distribution` module that defines
    the action distribution based on this latent representation.

    The class provides different forward methods for training (`forward`),
    exploration (`explore`), and deployment (`act`).

    Args:
        backbone (Module):
            The module for feature extraction.
        distribution (Distribution):
            The distribution module for exploration.
    """

    Factory = ActorFactory

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
        self.backbone: Module = backbone.rnn_compatible()
        self.distribution: Distribution = distribution
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
        """Main forward pass for the actor, dispatching to specific
        implementations.

        This method acts as a router to different functionalities based on the
        `forward_type` argument.
        """
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
    ) -> tuple[NestedTensor, tuple[Tensor, Tensor], Memory]:
        """Generates an action for exploration.

        This method is typically used during training to collect experience. It
        returns the parameters of the action distribution, the sampled action,
        and its log probability.

        Args:
            observation (Tensor):
                The input observation from the environment.
            memory (Memory, optional):
                The recurrent state for the backbone. Defaults to None.
            deterministic (bool, optional):
                If True, returns the mean of the distribution as the action
                instead of sampling. Defaults to False.
            backbone_kwargs (dict | None, optional):
                Additional keyword arguments for the backbone's forward pass.
                Defaults to None.
            distribution_kwargs (dict | None, optional):
                Additional keyword arguments for the distribution's forward
                pass. Defaults to None.

        Returns:
            - action_dist (NestedTensor):
                Distribution parameters.
            - action (tuple[Tensor, Tensor]):
                A tuple of (sampled_action, log_probability).
            - memory (Memory):
                The updated recurrent state.
        """
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
        """Generates an action for interacting with the environment.

        This is a simplified version of `explore` intended for deployment or
        evaluation, returning only the action and the updated memory state.

        Args:
            observation (Tensor):
                The input observation from the environment.
            memory (Memory, optional):
                The recurrent state for the backbone. Defaults to None.
            deterministic (bool, optional):
                If True, returns the mean of the distribution as the action
                instead of sampling. Defaults to False.
            backbone_kwargs (dict | None, optional):
                Additional keyword arguments for the backbone's forward pass.
                Defaults to None.
            distribution_kwargs (dict | None):
                Additional keyword arguments for the distribution's forward
                pass. Defaults to None.

        Returns:
            - action (Tensor):
                A tuple of (sampled_action, log_probability).
            - memory (Memory):
                The updated recurrent state.
        """
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
    ) -> tuple[NestedTensor, Memory]:
        latent, memory = self.backbone(
            observation,
            memory=memory,
            done=done,
            **(backbone_kwargs or {}),
        )

        dist_params = self.distribution(
            latent,
            observation=observation,
            **(distribution_kwargs or {}),
        )

        self.intermediate_repr["backbone.output"] = latent
        self.intermediate_repr.update(prefix_dict_keys(self.backbone.intermediate_repr, "backbone."))
        self.intermediate_repr.update(prefix_dict_keys(self.distribution.intermediate_repr, "distribution."))
        return dist_params, memory

    def _explore_impl(
        self,
        observation: Tensor,
        memory: Memory = None,
        deterministic: bool = False,
        backbone_kwargs: dict | None = None,
        distribution_kwargs: dict | None = None,
    ) -> tuple[NestedTensor, tuple[Tensor, Tensor], Memory]:
        latent, memory = self.backbone(
            observation,
            memory=memory,
            **(backbone_kwargs or {}),
        )
        if deterministic:
            dist_params = self.distribution(
                latent,
                observation=observation,
                **(distribution_kwargs or {}),
            )
            action = self.distribution.determine(
                latent,
                observation=observation,
                **(distribution_kwargs or {}),
            )
            logp = self.distribution.compute_logp(dist_params, action)
        else:
            dist_params, (action, logp) = self.distribution.sample(
                latent,
                observation=observation,
                **(distribution_kwargs or {}),
            )

        self.intermediate_repr["backbone.output"] = latent
        self.intermediate_repr.update(prefix_dict_keys(self.backbone.intermediate_repr, "backbone."))
        self.intermediate_repr.update(prefix_dict_keys(self.distribution.intermediate_repr, "distribution."))
        return dist_params, (action, logp), memory

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

    def compute_logp(self, dist_params: NestedTensor, action):
        return self.distribution.compute_logp(dist_params, action)

    def compute_entropy(self, dist_params: NestedTensor):
        return self.distribution.compute_entropy(dist_params)

    def compute_kl_div(self, dist_params1: NestedTensor, dist_params2: NestedTensor):
        return self.distribution.compute_kl_div(dist_params1, dist_params2)

    def step_memory(self, observation: Tensor, memory: Memory = None, **kwargs):
        return self.backbone.step_memory(observation, memory, **kwargs)

    def reset_memory(self, memory: Memory, done: Slice | Tensor | None = None):
        self.backbone.reset_memory(memory, done)

    def set_distribution_std(self, action_std):
        if hasattr(self.distribution, "set_std"):
            self.distribution.set_std(action_std)

    def clamp_distribution_std(self, lb=None, ub=None, indices=slice(None)):
        if hasattr(self.distribution, "clamp_std"):
            self.distribution.clamp_std(lb, ub, indices)
