from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any, Generic, TypeVar, overload

import numpy as np
import torch
from torch import nn

import cusrl
from cusrl.module.module import ModuleType
from cusrl.template.environment import EnvironmentSpec
from cusrl.utils import Metrics, distributed
from cusrl.utils.typing import Array, NestedArray, NestedTensor, Observation, Reward, State, Terminated, Truncated

__all__ = ["Agent", "AgentType", "AgentFactory"]


AgentType = TypeVar("AgentType", bound="Agent")


class AgentFactory(ABC, Generic[AgentType]):
    @abstractmethod
    def __call__(
        self,
        observation_dim: int,
        action_dim: int,
        state_dim: int | None = None,
        parallelism: int | None = None,
        environment_spec: EnvironmentSpec | None = None,
    ) -> AgentType:
        raise NotImplementedError

    def from_environment(self, environment: "cusrl.Environment") -> AgentType:
        return self(
            environment.observation_dim,
            environment.action_dim,
            environment.state_dim,
            environment.num_instances,
            environment.spec,
        )


class Agent(ABC):
    """Abstract base class for all reinforcement learning agents.

    This class defines the standard interface for an agent that interacts with an
    environment. It provides a framework for acting, processing environment steps,
    and updating internal models. Subclasses are required to implement the
    `act`, `step`, and `update` methods.

    The class also provides utilities for checkpoint management (saving and loading),
    device placement, mixed-precision training, and statistics tracking.

    Class Attributes:
        Factory (AgentFactory):
            A factory class used to create instances of the agent.
        MODULES (list[str]):
            A list of attribute names that correspond to `torch.nn.Module` instances.
            These modules will be automatically handled by methods like `state_dict`,
            `load_state_dict`, and `setup_module`.
        OPTIMIZERS (list[str]): A list of attribute names that correspond to
            `torch.optim.Optimizer` instances. These optimizers will be automatically
            handled by `state_dict` and `load_state_dict`.

    Args:
        observation_dim (int):
            The dimensionality of the observation space.
        action_dim (int):
            The dimensionality of the action space.
        state_dim (int | None):
            The dimensionality of the state space, defaults to `observation_dim` if None.
        parallelism (int | None):
            The number of parallel environments the agent is expected to interact with.
        environment_spec (EnvironmentSpec | None):
            Specifications of the environment.
        num_steps_per_update (int):
            The number of environment steps before triggering an update.
        name (str):
            The name of the agent.
        device (torch.device | str | None):
            The device (e.g., "cpu", "cuda") on which to place tensors and models.
        compile (bool):
            If True, `torch.compile` will be used on the modules to optimize performance.
        autocast (bool | torch.dtype):
            Enables automatic mixed precision. If True, defaults to `torch.float16`.
            Can be set to a specific `torch.dtype`. If False, mixed precision is disabled.
    """

    Factory = AgentFactory
    MODULES: list[str] = []
    OPTIMIZERS: list[str] = []

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        state_dim: int | None,
        parallelism: int | None,
        environment_spec: EnvironmentSpec | None,
        num_steps_per_update: int,
        name: str = "Agent",
        device: torch.device | str | None = None,
        compile: bool = False,
        autocast: bool | torch.dtype = False,
    ):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.has_state = state_dim is not None
        self.state_dim = state_dim or observation_dim
        self.parallelism = parallelism
        self.environment_spec = environment_spec or EnvironmentSpec()

        self.num_steps_per_update = num_steps_per_update
        self.name = name
        self.device = cusrl.device(device)
        self.compile = compile
        self.autocast_enabled = autocast is not None and autocast is not False
        self.dtype = autocast if isinstance(autocast, torch.dtype) else (torch.float16 if autocast else torch.float32)
        self.inference_mode = False
        self.deterministic = False

        self.transition = {}
        self.metrics = Metrics()
        self.iteration = 0
        self.step_index = 0

    @abstractmethod
    def act(self, observation: Observation, state: State = None) -> Array:
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        next_observation: Observation,
        reward: Reward,
        terminated: Terminated,
        truncated: Truncated,
        next_state: State = None,
        **kwargs: NestedArray,
    ) -> bool:
        """Returns True if the agent is ready to update."""
        if self.inference_mode:
            return False
        self.step_index += 1
        return self.step_index == self.num_steps_per_update

    @abstractmethod
    def update(self) -> dict[str, float]:
        self.step_index = 0
        self.iteration += 1
        metrics = self.metrics.summary(self.name)
        self.metrics.clear()
        return metrics

    def set_inference_mode(self, mode: bool = True, deterministic: bool | None = True):
        self.inference_mode = mode
        if deterministic is not None:
            self.deterministic = mode and deterministic

    def set_iteration(self, iteration: int):
        if iteration < 0:
            raise ValueError("Iteration must be non-negative.")
        self.iteration = iteration

    def to_tensor(self, input: Any) -> torch.Tensor:
        tensor = torch.as_tensor(input, device=self.device)
        if tensor is input:
            tensor = tensor.clone()
        return tensor

    @overload
    def to_nested_tensor(self, input: None) -> None: ...
    @overload
    def to_nested_tensor(self, input: Array) -> torch.Tensor: ...
    @overload
    def to_nested_tensor(self, input: tuple[NestedArray, ...] | list[NestedArray]) -> tuple[NestedTensor, ...]: ...
    @overload
    def to_nested_tensor(self, input: Mapping[str, NestedArray]) -> dict[str, NestedTensor]: ...

    def to_nested_tensor(self, input):
        if input is None:
            return None
        if isinstance(input, (tuple, list)):
            return tuple(self.to_nested_tensor(i) for i in input)
        if isinstance(input, Mapping):
            return {k: self.to_nested_tensor(v) for k, v in input.items()}
        return self.to_tensor(input)

    def setup_module(self, module: ModuleType) -> ModuleType:
        # Can also return a DistributedDataParallel instance with the module wrapped
        module = module.to(device=self.device)
        if distributed.enabled():
            module = distributed.make_distributed(module)
        return module

    def record(self, **kwargs):
        self.metrics.record(**kwargs)

    def state_dict(self):
        state_dict = {}
        for name in self.MODULES:
            if (module := getattr(self, name, None)) is not None:
                state_dict[name] = module.state_dict()
        for name in self.OPTIMIZERS:
            if (optim := getattr(self, name, None)) is not None:
                state_dict[name] = optim.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]):
        keys = set(state_dict.keys())
        for module_name in self.MODULES + self.OPTIMIZERS:
            module: nn.Module | None = getattr(self, module_name, None)
            if module is not None:
                if (state := state_dict.get(module_name)) is not None:
                    keys.discard(module_name)
                    try:
                        module.load_state_dict(state)
                    except (RuntimeError, ValueError) as error:
                        self.warn(f"Mismatched state_dict for '{module_name}': {error}")
                        continue
                else:
                    self.warn(f"Missing state_dict for '{module_name}'.")
        if keys:
            self.warn(f"Unused state_dict keys: {keys}.")

    def export(self, output_dir, **kwargs):
        pass

    @classmethod
    def warn(cls, info_str):
        distributed.print_once(f"\033[1;33mAgent: {info_str}\033[0m")

    @contextmanager
    def autocast(self):
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=self.autocast_enabled,
        ):
            yield

    def _train_mode(self, mode: bool = True):
        for name in self.MODULES:
            if (module := getattr(self, name, None)) is not None:
                module.train(mode)

    @classmethod
    def _decorator_update__set_to_training_mode(cls, update_method):
        def wrapped_update(self):
            self._train_mode(True)
            result = update_method(self)
            self._train_mode(False)
            return result

        return wrapped_update

    @classmethod
    def _decorator_act__preserve_io_format(cls, act_method):
        def wrapped_act(self, observation: Array, state: Array | None = None):
            action: torch.Tensor = act_method(self, observation, state)
            if isinstance(observation, np.ndarray):
                action_numpy: np.ndarray = action.cpu().numpy()
                if np.issubdtype(action_numpy.dtype, np.floating):
                    action_numpy = action_numpy.astype(dtype=observation.dtype)
                return action_numpy

            dtype = observation.dtype if torch.is_floating_point(action) else None
            return action.to(device=observation.device, dtype=dtype)

        return wrapped_act
