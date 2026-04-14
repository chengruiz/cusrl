from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

__all__ = ["OptimizerFactory"]


class OptimizerFactory:
    """Builds a PyTorch optimizer with per-prefix parameter overrides.

    ``defaults`` defines the base optimizer kwargs. They are passed to the
    optimizer constructor and also used for parameters that do not match any
    configured prefix. Prefix groups can be supplied through ``param_groups``
    or ``**kwargs``; the two mappings are merged and keyword arguments win on
    duplicate prefixes. Prefixes are sorted by length so the most specific
    match is applied first.

    A parameter matches a prefix when its name is exactly that prefix or begins
    with ``"{prefix}."``. Empty prefixes are not allowed. Unmatched parameters
    use ``defaults`` directly. Parameters with ``requires_grad=False`` are
    skipped. ``param_filter`` can be used to keep only parameters whose
    names match one of the configured prefixes before grouping.

    Args:
        cls (str | type[Optimizer]):
            Optimizer class (for example, ``torch.optim.Adam``) or the name of
            a class exposed from ``torch.optim``.
        defaults (dict[str, Any] | None, optional):
            Base optimizer keyword arguments. These are passed to the optimizer
            constructor and reused for unmatched parameters.
        param_groups (dict[str, Any] | None, optional):
            Mapping from parameter-name prefix to optimizer keyword arguments
            for that group. Use this when a prefix is not a valid Python
            keyword argument name. If ``None``, all parameters use ``defaults``.
            Defaults to ``None``.
        param_filter (Sequence[str] | None, optional):
            Optional parameter-name prefixes used to keep only a subset of the
            parameters before grouping. If ``None``, all parameters are kept.
            Defaults to ``None``.
        **kwargs (dict[str, Any]):
            Additional prefix groups provided as keyword arguments. These are
            merged with ``param_groups`` and override duplicate prefixes.
    """

    def __init__(
        self,
        cls: str | type[Optimizer],
        defaults: dict[str, Any] | None = None,
        param_groups: dict[str, Any] | None = None,
        param_filter: Sequence[str] | None = None,
        **kwargs: dict[str, Any],
    ):
        self.cls = cls
        self.defaults = defaults or {}
        if param_filter is None:
            self.param_filter = None
        else:
            param_filter = tuple(param_filter)
            for prefix in param_filter:
                if not prefix:
                    raise ValueError("Empty prefixes are not allowed in 'param_filter'")
            self.param_filter = tuple(sorted(param_filter, key=len, reverse=True))

        param_groups = (param_groups or {}) | kwargs
        for prefix in param_groups.keys():
            if not prefix:
                raise ValueError("Empty prefixes are not allowed; use the default group instead")
        # Sort by length of prefix
        self.param_groups = dict(sorted(param_groups.items(), key=lambda x: len(x[0]), reverse=True))

    def __call__(self, named_parameters: Iterable[tuple[str, nn.Parameter]]) -> Optimizer:
        """Instantiates the configured optimizer from named parameters.

        Trainable parameters are grouped by their most specific matching
        prefix. Each emitted parameter group contains ``params`` and
        ``param_names`` alongside the resolved optimizer kwargs for that group.
        Groups with no trainable parameters are omitted.

        Args:
            named_parameters (Iterable[tuple[str, nn.Parameter]]):
                Iterable of ``(name, parameter)`` pairs, typically from
                ``module.named_parameters()``.

        Returns:
            Optimizer:
                An initialized optimizer built from the resolved parameter
                groups and ``defaults``.
        """
        optim_cls: type[Optimizer] = getattr(torch.optim, self.cls) if isinstance(self.cls, str) else self.cls
        param_groups: dict[str, dict[str, Any]] = {}
        for name, param in named_parameters:
            if not param.requires_grad:
                continue
            if self.param_filter is not None and not self._get_matched_prefix(name, self.param_filter):
                continue
            prefix = self._get_matched_prefix(name, self.param_groups)
            param_group = param_groups.get(prefix)
            if param_group is None:
                params = self.param_groups.get(prefix, self.defaults)
                param_group = {"param_names": [], "params": [], **params}
                param_groups[prefix] = param_group
            param_group["param_names"].append(name)
            param_group["params"].append(param)
        if not param_groups:
            raise ValueError("No trainable parameters matched the optimizer filter")
        return optim_cls(param_groups.values(), **self.defaults)

    @staticmethod
    def _get_matched_prefix(name, prefixes) -> str:
        """Returns the most specific configured prefix for ``name``."""
        matched_prefix = ""
        for prefix in prefixes:
            if name == prefix or name.startswith(f"{prefix}."):
                matched_prefix = prefix
                break
        return matched_prefix
