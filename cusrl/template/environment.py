from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Generic, Optional, TypeAlias

import gymnasium as gym

from cusrl.utils.typing import Array, ArrayType, Nested, Slice, StateType

if TYPE_CHECKING:
    from cusrl.hook.symmetry import SymmetryDefLike

__all__ = [
    "Environment",
    "EnvironmentFactory",
    "EnvironmentSpec",
    "get_done_indices",
    "update_observation_and_state",
]


class EnvironmentSpec:
    """A class encapsulates environment-specific specifications and properties.

    This class stores parameters that define environment behavior, statistical
    properties, transformation capabilities, and other environment
    characteristics.

    Attributes:
        # Basic properties
        num_instances (int):
            Number of instances in the environment.
        observation_dim (int):
            Dimension of the observation space.
        action_dim (int):
            Dimension of the action space.
        state_dim (int | None, optional):
            Dimension of the state space. Defaults to None.
        reward_dim (int):
            The dimension of the reward. Defaults to 1.

        # Additional properties
        action_space (gym.spaces.Space | None):
            The action space of the environment. If provided, this space defines
            the valid actions that can be taken in the environment.
        autoreset (bool):
            Whether the environment automatically resets itself on terminal
            states inside `Environment.step`.
        environment_instance (Environment | None):
            The environment instance associated with the specification. It is
            recommended to define environment specifications in this class as
            interfaces rather than directly using the environment attributes and
            methods.
        final_state_is_missing (bool):
            Whether the environment omits the final state of an episode.
        observation_space (gym.spaces.Space | None):
            The observation space of the environment. If provided, this space
            defines the valid observations that can be received from the
            environment.
        timestep (float | None):
            The time duration for one environment step.

        # Symmetry transformations
        mirror_action (SymmetryDefLike | None):
            Definition for action symmetry transformations.
        mirror_observation (SymmetryDefLike | None):
            Definition for observation symmetry transformations.
        mirror_state (SymmetryDefLike | None):
            Definition for state symmetry transformations.

        # Predefined statistics
        action_denormalization (tuple[Array, Array] | None):
            Tuple of arrays (scale, shift) used for denormalizing the action
            within the environment. If provided, these statistics are applied as
            an element-wise affine layer as `action = original_action * scale
            + shift` appended to the actor upon export.
        observation_normalization: (tuple[Array, Array] | None):
            Tuple of arrays (scale, shift) used for normalizing observation
            within the environment. If provided, these statistics are applied as
            an element- wise affine layer as `observation =
            original_observation - shift) / scale` prepended to the actor upon
            export.
        state_normalization (tuple[Array, Array] | None):
            Tuple of arrays (scale, shift) used for normalizing state within the
            environment. If provided, these statistics are applied as an
            element-wise affine layer as `state = (original_state - shift) /
            scale` prepended to the critic upon export. (not implemented yet)

        # State/observation relationships
        observation_is_subset_of_state (Array | Slice | None):
            Definition of the one-to-one correspondence relationship from state
            to observation.

        # Statistical grouping
        observation_stat_groups (Sequence[tuple[int, int]]):
            Sequence of (start_idx, end_idx) pairs defining groups of
            observation dimensions that share statistical properties.
        state_stat_groups (Sequence[tuple[int, int]]):
            Sequence of (start_idx, end_idx) pairs defining groups of state
            dimensions that share statistical properties.

        # Imitation
        demonstration_sampler (Callable[[int], Array] | None):
            A callable that samples demonstrations from the environment. It
            should accept an integer representing the number of demonstrations
            to sample and return a demonstration array.

        extras (dict):
            Dictionary containing additional environment-specific properties.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        action_denormalization: tuple[Array, Array] | None = None,
        action_space: gym.spaces.Space | None = None,
        autoreset: bool = False,
        demonstration_sampler: Callable[[int], Array] | None = None,
        environment_instance: Optional["Environment"] = None,
        final_state_is_missing: bool = False,
        mirror_action: Optional["SymmetryDefLike"] = None,
        mirror_observation: Optional["SymmetryDefLike"] = None,
        mirror_state: Optional["SymmetryDefLike"] = None,
        num_instances: int = 1,
        observation_is_subset_of_state: Array | Slice | None = None,
        observation_stat_groups: Sequence[tuple[int, int]] = (),
        observation_normalization: tuple[Array, Array] | None = None,
        observation_space: gym.spaces.Space | None = None,
        reward_dim: int = 1,
        state_dim: int | None = None,
        state_stat_groups: Sequence[tuple[int, int]] = (),
        state_normalization: tuple[Array, Array] | None = None,
        timestep: float | None = None,
        **kwargs,
    ):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.action_denormalization = action_denormalization
        self.action_space = action_space
        self.autoreset = autoreset
        self.demonstration_sampler = demonstration_sampler
        self.environment_instance = environment_instance
        self.final_state_is_missing = final_state_is_missing
        self.mirror_action = mirror_action
        self.mirror_observation = mirror_observation
        self.mirror_state = mirror_state
        self.num_instances = num_instances
        self.observation_is_subset_of_state = observation_is_subset_of_state
        self.observation_stat_groups = tuple(observation_stat_groups)
        self.observation_normalization = observation_normalization
        self.observation_space = observation_space
        self.reward_dim = reward_dim
        self.state_dim = state_dim
        self.state_stat_groups = tuple(state_stat_groups)
        self.state_normalization = state_normalization
        self.timestep = timestep

        if "action_stats" in kwargs:
            raise ValueError("'action_stats' is removed. Use 'action_denormalization' instead.")
        if "action_normalization" in kwargs:
            raise ValueError("'action_normalization' is removed. Use 'action_denormalization' instead.")
        if "observation_stats" in kwargs:
            raise ValueError("'observation_stats' is removed. Use 'observation_normalization' instead.")
        if "observation_denormalization" in kwargs:
            raise ValueError("'observation_denormalization' is removed. Use 'observation_normalization' instead.")
        if "state_stats" in kwargs:
            raise ValueError("'state_stats' is removed. Use 'state_normalization' instead.")
        if "state_denormalization" in kwargs:
            raise ValueError("'state_denormalization' is removed. Use 'state_normalization' instead.")
        self.extras = kwargs

    def __getattr__(self, key: str):
        return self.extras[key]

    def get(self, key: str, default=None):
        if key in self.extras:
            return self.extras[key]
        return self.__dict__.get(key, default)

    def override(self, key: str, value: Any):
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.extras[key] = value


EnvironmentFactory: TypeAlias = Callable[[], "Environment"]


class Environment(ABC, Generic[ArrayType]):
    """Environment class for defining the interface of an environment.

    Args:
        observation_dim (int):
            Dimension of the observation space.
        action_dim (int):
            Dimension of the action space.
        num_instances (int):
            Number of instances in the environment.
        state_dim (int | None, optional):
            Dimension of the state space. Defaults to None.
        **kwargs:
            Additional properties of the environment. See `EnvironmentSpec`
            for details.
    """

    Factory = EnvironmentFactory
    Spec = EnvironmentSpec

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        action_denormalization: tuple[Array, Array] | None = None,
        action_space: gym.spaces.Space | None = None,
        autoreset: bool = False,
        demonstration_sampler: Callable[[int], Array] | None = None,
        final_state_is_missing: bool = False,
        mirror_action: Optional["SymmetryDefLike"] = None,
        mirror_observation: Optional["SymmetryDefLike"] = None,
        mirror_state: Optional["SymmetryDefLike"] = None,
        num_instances: int = 1,
        observation_is_subset_of_state: Array | Slice | None = None,
        observation_stat_groups: Sequence[tuple[int, int]] = (),
        observation_normalization: tuple[Array, Array] | None = None,
        observation_space: gym.spaces.Space | None = None,
        reward_dim: int = 1,
        state_dim: int | None = None,
        state_stat_groups: Sequence[tuple[int, int]] = (),
        state_normalization: tuple[Array, Array] | None = None,
        timestep: float | None = None,
        **kwargs: Any,
    ):
        self.num_instances = num_instances
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.spec = EnvironmentSpec(
            observation_dim=observation_dim,
            action_dim=action_dim,
            action_denormalization=action_denormalization,
            action_space=action_space,
            autoreset=autoreset,
            environment_instance=self,
            demonstration_sampler=demonstration_sampler,
            final_state_is_missing=final_state_is_missing,
            mirror_action=mirror_action,
            mirror_observation=mirror_observation,
            mirror_state=mirror_state,
            num_instances=num_instances,
            observation_is_subset_of_state=observation_is_subset_of_state,
            observation_stat_groups=observation_stat_groups,
            observation_normalization=observation_normalization,
            observation_space=observation_space,
            reward_dim=reward_dim,
            state_dim=state_dim,
            state_stat_groups=state_stat_groups,
            state_normalization=state_normalization,
            timestep=timestep,
            **kwargs,
        )

    @abstractmethod
    def reset(self, *, indices: ArrayType | Slice | None = None) -> tuple[
        ArrayType,
        ArrayType | None,
        dict[str, Nested[ArrayType]],
    ]:
        """Resets the environment.

        Args:
            indices (ArrayType | Slice | None, optional):
                Indices of instances to reset. If None, resets all instances.

        Returns:
        - observation (ArrayType):
            Observations of reset instances with shape `[Ni, Do]`, where `Ni` is
            the number of reset instances and `Do` is the observation dimension.
        - state (ArrayType | None):
            States of reset instances with shape `[Ni, Ds]`, where `Ds` is the
            state dimension. If the environment does not have privileged
            observations, the state should be None, which means the states are
            equal to the observations.
        - info (dict[str, Nested[ArrayType]]):
            Additional information dict for reset instances as named arrays.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action: ArrayType) -> tuple[
        ArrayType,
        ArrayType | None,
        ArrayType,
        ArrayType,
        ArrayType,
        dict[str, Nested[ArrayType]],
    ]:
        """Performs one environment step.

        Args:
            action (ArrayType):
                Actions of all instances with shape `[N, Da]`, where `N` is
                the number of instances and `Da` is the action dimension.

        Returns:
            - next_observation (ArrayType):
                Next observations of all instances with shape `[N, Do]`, where
                `Do` is the observation dimension.
            - next_state (ArrayType | None):
                Next states of all instances with shape `[N, Ds]`, where `Ds` is
                the state dimension; or None if equal to the observations.
            - reward (ArrayType):
                Rewards of all instances with shape `[N, Dr]`, where `Dr` is the
                reward dimension.
            - terminated (ArrayType):
                Boolean array indicating terminal states, with shape `[N, 1]`.
            - truncated (ArrayType):
                Boolean array indicating truncated states, with shape `[N, 1]`.
            - info (dict[str, Nested[ArrayType]]):
                Additional information dict for the step as named arrays, with
                shape `[N, Dk]`, where `Dk` is the dimension of the a specific
                information array.
        """
        raise NotImplementedError

    def get_metrics(self) -> dict[str, float]:
        """Gets metrics of the environment as a dictionary."""
        return {}

    def state_dict(self) -> dict[str, Any]:
        """Gets the state of the environment as a dictionary."""
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Loads the state of the environment from a dictionary."""
        pass


def get_done_indices(terminated: ArrayType, truncated: ArrayType) -> list[int]:
    done = terminated | truncated
    indices = done.squeeze(-1).nonzero()
    if isinstance(indices, tuple):  # for np.nonzero
        indices = indices[0]
    indices = indices.reshape(-1)  # for torch.nonzero
    return indices.tolist()


def update_observation_and_state(
    last_observation: ArrayType,
    last_state: StateType,
    indices: ArrayType | Slice,
    init_observation: ArrayType,
    init_state: StateType,
) -> tuple[ArrayType, StateType]:
    # If the complete observation of all instances is returned
    if init_observation.shape == last_observation.shape:
        return init_observation, init_state
    # Replace the observation and state of the reset instances
    last_observation[indices] = init_observation
    if last_state is not None:
        last_state[indices] = init_state
    return last_observation, last_state
