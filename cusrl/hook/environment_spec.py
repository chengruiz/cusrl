from cusrl.template import Hook

__all__ = ["EnvironmentSpecOverride"]


class EnvironmentSpecOverride(Hook):
    """Overrides attributes of the agent's environment specification.

    This hook allows for modifying the `environment_spec` of an agent before
    its initialization. The desired overrides are provided as keyword arguments
    during the hook's instantiation. These overrides are applied during the
    `pre_init` phase of the agent's lifecycle.

    Args:
        **kwargs:
            Arbitrary keyword arguments where each key is the name of the
            attribute to override in the `environment_spec` and the value is the
            new value for that attribute.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.overrides = kwargs

    def pre_init(self, agent):
        super().pre_init(agent)
        for key, value in self.overrides.items():
            agent.environment_spec.override(key, value)
