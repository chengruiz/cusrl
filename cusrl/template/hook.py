from collections.abc import Iterable, Iterator, Mapping
from typing import Any, Generic

import torch
from torch import nn
from typing_extensions import Self

import cusrl
from cusrl.module import GraphBuilder
from cusrl.template.agent import AgentType
from cusrl.template.buffer import Buffer
from cusrl.utils import distributed
from cusrl.utils.str_utils import camel_to_snake
from cusrl.utils.typing import NestedTensor

__all__ = ["Hook", "HookComposite"]


class Hook(Generic[AgentType]):
    """A component that extends an agent's functionality.

    Hooks are executed at specific points in the agent's lifecycle, such as
    before and after an action is taken, or before and after an update is
    performed.
    """

    def __init__(self):
        """Initializes the hook."""
        self.agent: AgentType
        self._modules: dict[str, nn.Module | None] = {}
        self._mutable: set[str] = set()
        self._name: str = camel_to_snake(self.__class__.__name__)
        self._active: bool = True

    @property
    def name(self) -> str:
        """Returns the name of the hook, which is the snake case of the class
        name by default."""
        return self._name

    @property
    def active(self) -> bool:
        """Returns whether the hook is active."""
        return self._active

    def name_(self, name: str) -> Self:
        """Overrides the default name of the hook.

        Args:
            name (str): The new name for the hook.
        """
        self._name = name
        return self

    def active_(self, active: bool) -> Self:
        """Overrides the default active state of the hook.

        Args:
            active (bool): The new active state for the hook.
        """
        self._active = active
        return self

    def register_module(self, name: str, module: nn.Module | None):
        """Registers a `torch.nn.Module` with the hook.

        The module will be moved to the agent's device and made distributed if
        needed. Its parameters and state_dict will be included in the
        `named_parameters` and `state_dict` methods.

        Args:
            name (str):
                The name of the module.
            module (nn.Module | None):
                The module to register.
        """
        if module is not None:
            module = self.agent.setup_module(module)
        setattr(self, name, module)
        self._modules[name] = module

    def register_mutable(self, name: str, value: Any):
        """Registers a mutable attribute with the hook.

        Mutable attributes can be updated during training via
        `update_attribute`.

        Args:
            name (str):
                The name of the attribute.
            value (Any):
                The value to assign.
        """
        setattr(self, name, value)
        self._mutable.add(name)

    def update_attribute(self, name: str, value: Any):
        """Updates a mutable attribute of the hook.

        Args:
            name (str):
                The name of the attribute to update.
            value (Any):
                The new value for the attribute.

        Raises:
            ValueError: If the attribute is not mutable.
        """
        if name not in self._mutable:
            raise ValueError(f"Attribute '{name}' is not mutable for hook {self.name}.")
        setattr(self, name, value)

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, nn.Parameter]]:
        """Returns an iterator over the hook's parameters, yielding both the
        name of the parameter and the parameter itself.

        Args:
            prefix (str, optional):
                The prefix to prepend to the parameter names.

        Yields:
            A tuple containing
                - name (str): The name of the parameter.
                - parameter (nn.Parameter): The parameter itself.
        """
        if prefix:
            prefix += "."
        for module_name, module in self._modules.items():
            if module is not None:
                yield from module.named_parameters(prefix=f"{prefix}{module_name}")

    def state_dict(self):
        """Returns a dictionary containing the state of the hook."""
        result = {}
        for module_name, module in self._modules.items():
            if module is not None:
                result[module_name] = module.state_dict()
        return result

    def load_state_dict(self, state_dict: Mapping[str, Any]):
        """Copies parameters from `state_dict` into the modules of the hook.

        Args:
            state_dict: A dictionary containing the state of the hook.
        """
        keys = set(state_dict.keys())
        for module_name, module in self._modules.items():
            if module is None:
                continue
            if module_name not in keys:
                self.warn(f"Missing state_dict for '{module_name}'.")
                continue
            keys.discard(module_name)
            try:
                module.load_state_dict(state_dict[module_name])
            except RuntimeError as error:
                self.warn(f"Mismatched state_dict for '{module_name}': {error}")
                continue

        if keys:
            self.warn(f"Unused state_dict keys: {keys}.")

    def compile(self):
        """Compiles the hook's modules using `torch.compile`."""
        for module in self._modules.values():
            if module is not None and hasattr(module, "compile"):
                module.compile()

    def train(self, mode: bool = True):
        """Sets the hook's modules to training mode.

        Args:
            mode: Whether to set training mode (`True`) or evaluation mode
            (`False`).
        """
        for module in self._modules.values():
            if module is not None and hasattr(module, "train"):
                module.train(mode)

    def eval(self):
        """Sets the hook's modules to evaluation mode."""
        self.train(False)

    def pre_init(self, agent: AgentType):
        """Called before the agent's modules are instantiated."""
        self.agent = agent

    def init(self):
        """Called after the agent's modules are instantiated.

        This is the proper place to initialize the hook's modules.
        """

    def post_init(self):
        """Called after the agent's modules and optimizers are fully
        initialized."""

    def pre_act(self, transition: dict[str, NestedTensor]):
        """Called before the agent's actor takes an action.

        Args:
            transition (dict[str, NestedTensor]):
                The transition dictionary, which contains the observation,
                state and other information.
        """

    def post_act(self, transition: dict[str, NestedTensor]):
        """Called after the agent's actor takes an action.

        Args:
            transition (dict[str, NestedTensor]):
                The transition dictionary, which contains the observation,
                state, action and other information.
        """

    def post_step(self, transition: dict[str, NestedTensor]):
        """Called after the agent takes a step in the environment.

        Args:
            transition:
                The transition dictionary, which contains the full transition
                information.
        """

    def pre_update(self, buffer: Buffer):
        """Called before the agent's update phase.

        Args:
            buffer (Buffer):
                The buffer containing the collected experience.
        """

    def objective(self, batch: dict[str, NestedTensor | Any]) -> torch.Tensor | None:
        """Defines the objective function for the agent's update.

        Args:
            batch (dict[str, NestedTensor | Any]):
                A batch of experience sampled from the buffer.

        Returns:
            loss (torch.Tensor | None):
                The computed loss tensor, which will be used to update the
                agent.
        """
        return None

    def pre_optim(self, optimizer: torch.optim.Optimizer):
        """Called before the optimizer's step.

        This is the proper place to perform gradient clipping or other gradient-
        based operations.
        """

    def post_optim(self):
        """Called after the optimizer's step."""

    def post_update(self):
        """Called after the agent's update phase."""

    def apply_schedule(self, iteration: int):
        """Applies a schedule based on the current iteration.

        This is the proper place to update learning rates or other
        hyperparameters.

        Args:
            iteration: The current training iteration.
        """

    def pre_export(self, graph: GraphBuilder):
        """Called before exporting the agent's model.

        Args:
            graph: The graph builder instance.
        """

    def post_export(self, graph: GraphBuilder):
        """Called after exporting the agent's model.

        Args:
            graph: The graph builder instance.
        """

    @classmethod
    def warn(cls, message):
        """Prints a warning message."""
        distributed.print_rank0(f"\033[1;31m{cls.__name__}: {message}\033[0m")


class HookComposite(Hook):
    """Wraps multiple hooks and executes them in sequence."""

    def __init__(self, hooks: Iterable[Hook]):
        super().__init__()
        self.hooks = tuple(hooks)
        self._named_hooks = {}
        for hook in self.hooks:
            if not isinstance(hook, Hook):
                raise TypeError(f"Expected 'Hook', got '{type(hook).__name__}'")
            if hook.name in self._named_hooks:
                raise RuntimeError(f"Hook '{hook.name}' already exists.")
            self._named_hooks[hook.name] = hook

    def __getitem__(self, name: str) -> Hook:
        """Returns the hook with the given name."""
        return self._named_hooks[name]

    def __iter__(self) -> Iterator[Hook]:
        yield from self.hooks

    def named_parameters(self, prefix: str = ""):
        if prefix and not prefix.endswith("."):
            prefix += "."
        for hook_name, hook in self._named_hooks.items():
            yield from hook.named_parameters(prefix=f"{prefix}{hook_name}")

    def state_dict(self):
        result = {}
        for hook_name, hook in self._named_hooks.items():
            if state_dict := hook.state_dict():
                result[hook_name] = state_dict
        return result

    def load_state_dict(self, state_dict: Mapping[str, Any]):
        keys = set(state_dict.keys())
        for hook_name, hook in self._named_hooks.items():
            if state := state_dict.get(hook_name):
                hook.load_state_dict(state)
            elif state := state_dict.get(hook_name := hook.__class__.__name__):  # For compatibility
                hook.load_state_dict(state)
            elif hook.state_dict():
                self.warn(f"Missing state_dict for '{hook.name}'.")
            keys.discard(hook_name)
        if keys:
            self.warn(f"Unused state_dict keys: {keys}.")

    def compile(self):
        for hook in self:
            hook.compile()

    def train(self, mode=True):
        for hook in self.active_hooks():
            hook.train(mode)

    def pre_init(self, agent: "cusrl.Agent"):
        super().pre_init(agent)
        for hook in self.active_hooks():
            hook.pre_init(agent)

    def init(self):
        for hook in self.active_hooks():
            hook.init()

    def post_init(self):
        for hook in self.active_hooks():
            hook.post_init()

    def pre_act(self, transition):
        for hook in self.active_hooks():
            hook.pre_act(transition)

    def post_act(self, transition):
        for hook in self.active_hooks():
            hook.post_act(transition)

    def post_step(self, transition):
        for hook in self.active_hooks():
            hook.post_step(transition)

    def pre_update(self, buffer):
        for hook in self.active_hooks():
            hook.pre_update(buffer)

    def objective(self, batch: dict[str, NestedTensor | Any]) -> torch.Tensor | None:
        objectives = []
        for hook in self.active_hooks():
            if (obj := hook.objective(batch)) is not None:
                objectives.append(obj)
        if objectives:
            return sum(objectives)
        return None

    def pre_optim(self, optimizer):
        for hook in self.active_hooks():
            hook.pre_optim(optimizer)

    def post_optim(self):
        for hook in self.active_hooks():
            hook.post_optim()

    def post_update(self):
        for hook in self.active_hooks():
            hook.post_update()

    def apply_schedule(self, iteration: int):
        for hook in self.active_hooks():
            hook.apply_schedule(iteration)

    def pre_export(self, graph: GraphBuilder):
        for hook in self:
            hook.pre_export(graph)

    def post_export(self, graph: GraphBuilder):
        for hook in self:
            hook.post_export(graph)

    def active_hooks(self) -> Iterator[Hook]:
        """Returns an iterator over the active hooks in the composite."""
        for hook in self:
            if hook.active:
                yield hook
