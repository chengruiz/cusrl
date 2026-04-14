from collections.abc import Sequence
from typing import Any

from cusrl.template.optimizer import OptimizerFactory

__all__ = ["AdamFactory", "AdamWFactory"]


class AdamFactory(OptimizerFactory):
    """Alias for :cls:`OptimizerFactory` with ``cls="Adam"``."""

    def __init__(
        self,
        defaults: dict[str, Any] | None = None,
        param_groups: dict[str, Any] | None = None,
        param_filter: Sequence[str] | None = None,
        **kwargs: dict[str, Any],
    ):
        super().__init__(
            cls="Adam",
            defaults=defaults,
            param_groups=param_groups,
            param_filter=param_filter,
            **kwargs,
        )


class AdamWFactory(OptimizerFactory):
    """Alias for :cls:`OptimizerFactory` with ``cls="AdamW"``."""

    def __init__(
        self,
        defaults: dict[str, Any] | None = None,
        param_groups: dict[str, Any] | None = None,
        param_filter: Sequence[str] | None = None,
        **kwargs: dict[str, Any],
    ):
        super().__init__(
            cls="AdamW",
            defaults=defaults,
            param_groups=param_groups,
            param_filter=param_filter,
            **kwargs,
        )
