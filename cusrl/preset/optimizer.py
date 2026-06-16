from collections.abc import Sequence
from typing import Any

from cusrl.template.optimizer import OptimizerFactory, OptimizerGroupOverride, OptimizerParamSelector

__all__ = ["AdamFactory", "AdamWFactory"]


class AdamFactory(OptimizerFactory):
    """Alias for :cls:`OptimizerFactory` with ``cls="Adam"``."""

    def __init__(
        self,
        defaults: dict[str, Any] | None = None,
        group_overrides: Sequence[OptimizerGroupOverride] | None = None,
        param_filter: str | Sequence[str] | OptimizerParamSelector | None = None,
    ):
        super().__init__(
            cls="Adam",
            defaults=defaults,
            group_overrides=group_overrides,
            param_filter=param_filter,
        )


class AdamWFactory(OptimizerFactory):
    """Alias for :cls:`OptimizerFactory` with ``cls="AdamW"``."""

    def __init__(
        self,
        defaults: dict[str, Any] | None = None,
        group_overrides: Sequence[OptimizerGroupOverride] | None = None,
        param_filter: str | Sequence[str] | OptimizerParamSelector | None = None,
    ):
        super().__init__(
            cls="AdamW",
            defaults=defaults,
            group_overrides=group_overrides,
            param_filter=param_filter,
        )
