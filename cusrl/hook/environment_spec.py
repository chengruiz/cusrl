from collections.abc import Callable
from typing import Any

from cusrl.template import Agent, Hook
from cusrl.template.environment import Environment

__all__ = ["EnvironmentSpecOverride", "DynamicEnvironmentSpecOverride"]


class EnvironmentSpecOverride(Hook):
    """Overrides attributes of the agent's environment specification.

    This hook allows for modifying the :attr:`environment_spec` of an agent
    before its initialization. The desired overrides are provided as keyword
    arguments during the hook's instantiation. These overrides are applied
    during the :func:`pre_init` phase of the agent's lifecycle.

    Args:
        **kwargs:
            Keyword arguments used to override attributes in the
            :attr:`environment_spec`. Each key represents the attribute's name,
            and the corresponding value is the new value to be set.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.overrides = kwargs

    def pre_init(self, agent: Agent):
        super().pre_init(agent)
        for key, value in self.overrides.items():
            agent.environment_spec.override(key, value)


class DynamicEnvironmentSpecOverride(Hook):
    """Dynamically overrides attributes of the agent's environment specification
    with the environment instance.

    This hook allows for modifying the :attr:`environment_spec` of an agent
    before its initialization. The desired overrides are generated dynamically
    based on the environment instance.

    Args:
        overrides_factory (Callable[[Environment], dict[str, Any]]):
            A factory function that generates the overrides based on the
            environment instance.
    """

    def __init__(self, overrides_factory: Callable[[Environment], dict[str, Any]]):
        super().__init__()
        self.overrides_factory = overrides_factory

    def pre_init(self, agent: Agent):
        super().pre_init(agent)
        if not agent.environment_spec.environment_instance:
            raise ValueError("Environment instance is not set in the agent's environment_spec.")
        overrides = self.overrides_factory(agent.environment_spec.environment_instance)
        for key, value in overrides.items():
            agent.environment_spec.override(key, value)
