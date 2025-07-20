from collections.abc import Iterable, Iterator
from typing import Any, Generic

import torch
from torch import nn

import cusrl
from cusrl.template.agent import AgentType
from cusrl.utils import distributed
from cusrl.utils.export import ExportGraph

__all__ = ["Hook", "HookComposite"]


class Hook(Generic[AgentType]):
    agent: AgentType
    MODULES: list[str] = []
    PARAMETERS: list[str] = []
    MUTABLE_ATTRS: list[str] = []
    active: bool = True

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, nn.Parameter]]:
        if prefix:
            prefix += "."
        for module_name in self.MODULES:
            if (module := getattr(self, module_name, None)) is not None:
                yield from module.named_parameters(prefix=f"{prefix}{module_name}")
        for param_name in self.PARAMETERS:
            if (param := getattr(self, param_name, None)) is not None:
                yield f"{prefix}{param_name}", param

    def state_dict(self):
        result = {}
        for module_name in self.MODULES:
            if (module := getattr(self, module_name, None)) is not None:
                result[module_name] = module.state_dict()
        for param_name in self.PARAMETERS:
            if (param := getattr(self, param_name, None)) is not None:
                result[param_name] = param
        return result

    def load_state_dict(self, state_dict):
        keys = set(state_dict.keys())
        for module_name in self.MODULES:
            module: nn.Module | None = getattr(self, module_name, None)
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

        for param_name in self.PARAMETERS:
            param: nn.Parameter | None = getattr(self, param_name, None)
            if param is None:
                continue
            if param_name not in keys:
                self.warn(f"Missing state_dict for '{param_name}'.")
                continue
            keys.discard(param_name)
            try:
                param[:] = state_dict[param_name]
            except RuntimeError as error:
                self.warn(f"Mismatched state_dict for '{param_name}': {error}")
                continue

        if keys:
            self.warn(f"Unused state_dict keys: {keys}.")

    def compile(self):
        for module_name in self.MODULES:
            if hasattr((module := getattr(self, module_name, None)), "compile"):
                module.compile()

    def train(self, mode: bool = True):
        for module_name in self.MODULES:
            if hasattr((module := getattr(self, module_name, None)), "train"):
                module.train(mode)

    def eval(self):
        self.train(False)

    def pre_init(self, agent: AgentType):
        self.agent = agent

    def init(self):
        pass

    def post_init(self):
        pass

    def pre_act(self, transition: dict[str, torch.Tensor | Any]):
        pass

    def post_act(self, transition: dict[str, torch.Tensor | Any]):
        pass

    def post_step(self, transition: dict[str, torch.Tensor | Any]):
        pass

    def pre_update(self, buffer: "cusrl.Buffer"):
        pass

    def objective(self, batch: dict[str, torch.Tensor | Any]) -> torch.Tensor | None:
        return None

    def pre_optim(self, optimizer: torch.optim.Optimizer):
        pass

    def post_update(self):
        pass

    def apply_schedule(self, iteration: int):
        pass

    def update_attribute(self, name, value):
        if name not in self.MUTABLE_ATTRS:
            raise ValueError(f"Attribute '{name}' is not mutable for hook {self.name}.")
        setattr(self, name, value)

    def pre_export(self, graph: ExportGraph):
        pass

    def post_export(self, graph: ExportGraph):
        pass

    @classmethod
    def warn(cls, info_str):
        distributed.print_once(f"\033[1;31m{cls.__name__}: {info_str}\033[0m")


class HookComposite(Hook):
    def __init__(self, hooks: Iterable[Hook]):
        self.hooks = tuple(hooks)
        self._named_hooks = {}
        for hook in self.hooks:
            if not isinstance(hook, Hook):
                raise TypeError(f"Expected 'Hook', got '{type(hook).__name__}'")
            if hook.name in self._named_hooks:
                raise RuntimeError(f"Hook '{hook.name}' already exists.")
            self._named_hooks[hook.name] = hook

    def __getitem__(self, item: str) -> Hook:
        return self._named_hooks[item]

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

    def load_state_dict(self, state_dict: dict):
        keys = set(state_dict.keys())
        for hook_name, hook in self._named_hooks.items():
            if state := state_dict.get(hook_name):
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

    def objective(self, batch: dict[str, Any]) -> torch.Tensor | None:
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

    def post_update(self):
        for hook in self.active_hooks():
            hook.post_update()

    def apply_schedule(self, iteration: int):
        for hook in self.active_hooks():
            hook.apply_schedule(iteration)

    def pre_export(self, graph: ExportGraph):
        for hook in self:
            hook.pre_export(graph)

    def post_export(self, graph: ExportGraph):
        for hook in self:
            hook.post_export(graph)

    def active_hooks(self) -> Iterator[Hook]:
        for hook in self:
            if hook.active:
                yield hook
