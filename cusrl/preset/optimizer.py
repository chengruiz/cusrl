from typing import Any

from cusrl.template.optimizer import OptimizerFactory

__all__ = ["AdamFactory", "AdamWFactory"]


class AdamFactory(OptimizerFactory):
    """Alias for :cls:`OptimizerFactory` with ``cls="Adam"``."""

    def __init__(self, defaults: dict[str, Any] | None = None, **optim_groups: dict[str, Any]):
        super().__init__(cls="Adam", defaults=defaults, **optim_groups)


class AdamWFactory(OptimizerFactory):
    """Alias for :cls:`OptimizerFactory` with ``cls="AdamW"``."""

    def __init__(self, defaults: dict[str, Any] | None = None, **optim_groups: dict[str, Any]):
        super().__init__(cls="AdamW", defaults=defaults, **optim_groups)
