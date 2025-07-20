from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Optional, TypeAlias

from cusrl.utils.typing import (
    Array,
    ArrayType,
    BoolArrayType,
    Info,
    Observation,
    Reward,
    Slice,
    State,
    StateType,
    Terminated,
    Truncated,
)

if TYPE_CHECKING:
    from cusrl.hook.symmetry import SymmetryDef

__all__ = ["Environment", "EnvironmentFactory", "EnvironmentSpec", "get_done_indices", "update_observation_and_state"]


class EnvironmentSpec:
    """A class encapsulates environment-specific specifications and properties.

    This class stores parameters that define environment behavior, statistical
    properties, transformation capabilities, and other environment characteristics.

    Attributes:
        autoreset (bool):
            Whether the environment automatically resets itself on terminal states
            inside `Environment.step`.
        final_state_is_missing (bool):
            Whether the environment omits the final state of an episode.
        reward_dim (int):
            The dimension of the reward. Defaults to 1.
        timestep (float | None):
            The time duration for one environment step.

        # Symmetry transformations
        mirror_action (SymmetryDef | None):
            Definition for action symmetry transformations.
        mirror_observation (SymmetryDef | None):
            Definition for observation symmetry transformations.
        mirror_state (SymmetryDef | None):
            Definition for state symmetry transformations.

        # Predefined statistics
        action_stats (tuple[Array, Array] | None):
            Tuple of arrays (mean, standard deviation) for scaling and biasing actions
            within the environment. If provided, these statistics are applied as a
            denormalization layer appended to the actor upon export.
        observation_stats (tuple[Array, Array] | None):
            Tuple of arrays (mean, standard deviation) used to scale and bias
            observations within the environment. If provided, these statistics are
            applied as a denormalization layer prepended to the actor upon export.
        state_stats (tuple[Array, Array] | None):
            Tuple of arrays (mean, standard deviation) used to scale and bias
            states within the environment. If provided, these statistics are applied
            as a denormalization layer prepended to the critic upon export.
            (not implemented yet)

        # State/observation relationships
        observation_is_subset_of_state (Array | Slice | None):
            Definition of the one-to-one correspondence relationship from state to
            observation.

        # Statistical grouping
        observation_stat_groups (Sequence[tuple[int, int]]):
            Sequence of (start_idx, end_idx) pairs defining groups of observation
            dimensions that share statistical properties.
        state_stat_groups (Sequence[tuple[int, int]]):
            Sequence of (start_idx, end_idx) pairs defining groups of state dimensions
            that share statistical properties.

        extras (dict): Dictionary containing additional environment-specific properties.
    """

    def __init__(
        self,
        *,
        action_stats: tuple[Array, Array] | None = None,
        autoreset: bool = False,
        final_state_is_missing: bool = False,
        mirror_action: Optional["SymmetryDef"] = None,
        mirror_observation: Optional["SymmetryDef"] = None,
        mirror_state: Optional["SymmetryDef"] = None,
        observation_is_subset_of_state: Array | Slice | None = None,
        observation_stat_groups: Sequence[tuple[int, int]] = (),
        observation_stats: tuple[Array, Array] | None = None,
        reward_dim: int = 1,
        state_stat_groups: Sequence[tuple[int, int]] = (),
        state_stats: tuple[Array, Array] | None = None,
        timestep: float | None = None,
        **kwargs,
    ):
        self.action_stats = action_stats
        self.autoreset = autoreset
        self.final_state_is_missing = final_state_is_missing
        self.mirror_action = mirror_action
        self.mirror_observation = mirror_observation
        self.mirror_state = mirror_state
        self.observation_is_subset_of_state = observation_is_subset_of_state
        self.observation_stat_groups = tuple(observation_stat_groups)
        self.observation_stats = observation_stats
        self.reward_dim = reward_dim
        self.state_stat_groups = tuple(state_stat_groups)
        self.state_stats = state_stats
        self.timestep = timestep
        self.extras = kwargs

    def __getattr__(self, key: str):
        return self.extras[key]

    def get(self, key: str, default=None):
        if key in self.extras:
            return self.extras[key]
        return self.__dict__.get(key, default)


EnvironmentFactory: TypeAlias = Callable[[], "Environment"]


class Environment(ABC):
    """Environment class for defining the interface of an environment.

    Args:
        num_instances (int):
            Number of instances in the environment.
        observation_dim (int):
            Dimension of the observation space.
        action_dim (int):
            Dimension of the action space.
        state_dim (int | None, optional):
            Dimension of the state space. Defaults to None.
        spec (dict | None | EnvironmentSpec, optional):
            Additional properties of the environment. Defaults to None.

    Key methods:
        reset(indices: Array | Slice | None = None) -> tuple[Observation, State, Info]:
            Abstract method to reset the environment. Must be implemented by subclasses.
        step(action: Array) -> tuple[Observation, State, Reward, Terminated, Truncated, Info]:
            Abstract method to take a step in the environment. Must be implemented by subclasses.
        get_metrics() -> dict[str, float]:
            Returns metrics as a dictionary.
        state_dict() -> dict[str, Array]:
            Returns the state of the environment as a dictionary.
        load_state_dict(state_dict: dict[str, Array]):
            Loads the state of the environment from a dictionary.
    """

    Factory = EnvironmentFactory
    Spec = EnvironmentSpec

    def __init__(
        self,
        num_instances: int,
        observation_dim: int,
        action_dim: int,
        state_dim: int | None = None,
        spec: dict | None | EnvironmentSpec = None,
    ):
        self.num_instances = num_instances
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        if not isinstance(spec, EnvironmentSpec):
            spec = EnvironmentSpec(**(spec or {}))
        self.spec: EnvironmentSpec = spec

    # fmt: off
    @abstractmethod
    def reset(self, *, indices: Array | Slice | None = None) -> tuple[
        Observation,  # [ N / Ni, Do ], f32 (observation of all or reset instances)
        State,        # [ N / Ni, Ds ], f32 (state of all or reset instances)
        Info,         # [ N / Ni, Dk ]
    ]:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Array) -> tuple[
        Observation,  # [ N, Do ], f32
        State,        # [ N, Ds ], f32
        Reward,       # [ N, Dr ], f32
        Terminated,   # [ N,  1 ], bool
        Truncated,    # [ N,  1 ], bool
        Info,         # [ N, Dk ]
    ]:
        raise NotImplementedError
    # fmt: on

    def get_metrics(self) -> dict[str, float]:
        return {}

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]):
        pass


def get_done_indices(terminated: BoolArrayType, truncated: BoolArrayType) -> list[int]:
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
