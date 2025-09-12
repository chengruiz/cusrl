from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any, Generic, TypeVar, overload

import numpy as np
import torch
from torch import nn
from typing_extensions import Self

import cusrl
from cusrl.template.environment import Environment
from cusrl.utils import Metrics, distributed
from cusrl.utils.typing import Array, ArrayType, ListOrTuple, Nested, NestedArray, NestedTensor

__all__ = ["Agent", "AgentType", "AgentFactory"]


AgentType = TypeVar("AgentType", bound="Agent")


class AgentFactory(ABC, Generic[AgentType]):
    def __init__(
        self,
        num_steps_per_update: int,
        name: str = "Agent",
        device: torch.device | str | None = None,
        compile: bool = False,
        autocast: bool | str | torch.dtype = False,
    ):
        self.num_steps_per_update = num_steps_per_update
        self.name = name
        self.device = device
        self.compile = compile
        if isinstance(autocast, str):
            dtype = getattr(torch, autocast, None)
            if dtype is None or not isinstance(dtype, torch.dtype):
                raise ValueError(f"Invalid autocast datatype '{autocast}'.")
            autocast = dtype
        self.autocast = autocast

    def override(self, **kwargs: Any) -> Self:
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid argument '{key}' for {self.__class__.__name__}.")
            setattr(self, key, value)
        return self

    @abstractmethod
    def __call__(self, environment_spec: Environment.Spec) -> AgentType:
        raise NotImplementedError

    def from_environment(self, environment: Environment) -> AgentType:
        return self(environment.spec)


_ModuleType = TypeVar("_ModuleType", bound=nn.Module)


class Agent(ABC):
    """Abstract base class for all reinforcement learning agents.

    This class defines the standard interface for an agent that interacts with
    an environment. It provides a framework for acting, processing environment
    steps, and updating internal models. Subclasses are required to implement
    the `act`, `step`, and `update` methods.

    The class also provides utilities for checkpoint management (saving and
    loading), device placement, mixed-precision training, and statistics
    tracking.

    Attributes:
        Factory (AgentFactory):
            A factory class used to create instances of the agent.
        MODULES (list[str]):
            A list of attribute names that correspond to `torch.nn.Module`
            instances. These modules will be automatically handled by methods
            like `state_dict`, `load_state_dict`, and `_set_training_mode`.
        OPTIMIZERS (list[str]):
            A list of attribute names that correspond to `torch.optim.Optimizer`
            instances. These optimizers will be automatically handled by
            `state_dict` and `load_state_dict`.

    Args:
        environment_spec (EnvironmentSpec):
            Specifications of the environment.
        num_steps_per_update (int):
            The number of environment steps before triggering an update.
        name (str):
            The name of the agent.
        device (torch.device | str | None):
            The device (e.g., "cpu", "cuda") on which to place tensors and
            models.
        compile (bool):
            If True, `torch.compile` will be used on the modules to optimize
            performance.
        autocast (bool | torch.dtype):
            Enables automatic mixed precision. If True, defaults to
            `torch.float16`. Can be set to a specific `torch.dtype`. If False,
            mixed precision is disabled.
    """

    Factory = AgentFactory
    MODULES: list[str] = []
    OPTIMIZERS: list[str] = []

    def __init__(
        self,
        environment_spec: Environment.Spec,
        num_steps_per_update: int,
        name: str = "Agent",
        device: torch.device | str | None = None,
        compile: bool = False,
        autocast: bool | torch.dtype = False,
    ):
        self.observation_dim = environment_spec.observation_dim
        self.action_dim = environment_spec.action_dim
        self.has_state = environment_spec.state_dim is not None
        self.state_dim = environment_spec.state_dim or self.observation_dim
        self.parallelism = environment_spec.num_instances
        self.environment_spec = environment_spec

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
    def act(self, observation: ArrayType, state: ArrayType | None = None) -> ArrayType:
        """Selects an action based on the current observation and state.

        This method is responsible for choosing an action from the agent's
        policy, given the current state of the environment.

        Args:
            observation (ArrayType):
                The current observation from the environment.
            state (ArrayType | None, optional):
                The current state from the environment (e.g. privileged
                observation). Defaults to None.

        Returns:
            action (ArrayType):
                The action to be taken in the environment.
        """
        ...

    @abstractmethod
    def step(
        self,
        next_observation: ArrayType,
        reward: ArrayType,
        terminated: ArrayType,
        truncated: ArrayType,
        next_state: ArrayType | None = None,
        **kwargs: Nested[ArrayType],
    ) -> bool:
        """Processes a single step of interaction with the environment.

        This method is called after the environment has executed an action. It
        is used to record the transition (observation, action, reward, etc.)
        and prepare for the next action or update.

        Args:
            next_observation (ArrayType):
                The observation received from the environment after taking the
                action.
            reward (ArrayType):
                The reward received from the environment.
            terminated (ArrayType):
                A boolean array indicating whether the episode has terminated.
            truncated (ArrayType):
                A boolean array indicating whether the episode has been
                truncated.
            next_state (ArrayType | None, optional):
                The next state from the environment. Defaults to None.
            **kwargs (Nested[ArrayType]):
                Additional data from the environment step.

        Returns:
            ready_for_update (bool):
                True if the agent is ready for an update, False otherwise.
        """
        if self.inference_mode:
            return False
        self.step_index += 1
        return self.step_index == self.num_steps_per_update

    @abstractmethod
    def update(self) -> dict[str, float]:
        """Performs an update of the agent's modules.

        This method is called when the agent has collected enough experience
        (i.e., `num_steps_per_update`). It triggers the learning process,
        updating the agent's parameters based on the collected data.

        Returns:
            metrics (dict[str, float]):
                A dictionary of metrics from the update step.
        """
        self.step_index = 0
        self.iteration += 1
        metrics = self.metrics.summary(self.name)
        self.metrics.clear()
        return metrics

    def set_inference_mode(self, mode: bool = True, deterministic: bool | None = True):
        """Sets the agent to inference mode. Mainly used for evaluation."""
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
    def to_nested_tensor(self, input: ListOrTuple[NestedArray]) -> tuple[NestedTensor, ...]: ...
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

    def setup_module(self, module: _ModuleType) -> _ModuleType:
        # Can also return a DistributedDataParallel instance with the module wrapped
        module = module.to(device=self.device)
        if distributed.enabled():
            module = distributed.make_distributed(module)
        return module

    def record(self, **kwargs):
        """Record metrics for the agent."""
        self.metrics.record(**kwargs)

    def state_dict(self):
        state_dict = {}
        for name in self.MODULES + self.OPTIMIZERS:
            if (module := getattr(self, name, None)) is not None:
                state_dict[name] = module.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]):
        keys = set(state_dict.keys())
        for name in self.MODULES + self.OPTIMIZERS:
            module: nn.Module | torch.optim.Optimizer | None = getattr(self, name, None)
            if module is None:
                continue
            if (state := state_dict.get(name)) is None:
                self.warn(f"Missing state_dict for '{name}'.")
                continue
            keys.discard(name)
            try:
                module.load_state_dict(state)
            except (RuntimeError, ValueError) as error:
                self.warn(f"Mismatched state_dict for '{name}': {error}")
        if keys:
            self.warn(f"Unused state_dict keys: {keys}.")

    def export(self, output_dir, **kwargs):
        pass

    @classmethod
    def warn(cls, info_str):
        distributed.print_rank0(f"\033[1;33mAgent: {info_str}\033[0m")

    @contextmanager
    def autocast(self):
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=self.autocast_enabled,
        ):
            yield

    def _set_training_mode(self, mode: bool = True):
        for name in self.MODULES:
            if (module := getattr(self, name, None)) is not None:
                module.train(mode)

    @classmethod
    def _decorator_update__set_training_mode(cls, update_method):
        def wrapped_update(self):
            self._set_training_mode(True)
            result = update_method(self)
            self._set_training_mode(False)
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
