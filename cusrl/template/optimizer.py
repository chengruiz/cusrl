from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

__all__ = ["OptimizerCollection", "OptimizerFactory", "build_optimizer"]


class OptimizerCollection:
    """A small optimizer-compatible wrapper around named optimizers.

    The wrapper exposes the subset of the :class:`torch.optim.Optimizer`
    interface used by agents and hooks while delegating all optimization work
    to its child optimizers.
    """

    def __init__(self, optimizers: Mapping[str, Optimizer]):
        if not optimizers:
            raise ValueError("At least one optimizer is required")

        self.optimizers = dict(optimizers)
        for name in self.optimizers:
            if not isinstance(name, str) or not name:
                raise ValueError("Optimizer names must be non-empty strings")

        self._tag_param_groups()
        self._validate_unique_parameters()

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        """Returns a flattened view of all child optimizer parameter groups."""
        return [param_group for optimizer in self.optimizers.values() for param_group in optimizer.param_groups]

    def zero_grad(self, *args, **kwargs):
        """Clears gradients for all child optimizers."""
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs):
        """Steps all child optimizers in insertion order."""
        for optimizer in self.optimizers.values():
            optimizer.step(*args, **kwargs)

    def state_dict(self) -> dict[str, Any]:
        """Returns child optimizer states keyed by optimizer name."""
        return {name: optimizer.state_dict() for name, optimizer in self.optimizers.items()}

    def load_state_dict(self, state_dict: Mapping[str, Any]):
        """Loads child optimizer states keyed by optimizer name."""
        expected_names = set(self.optimizers)
        found_names = set(state_dict)
        if expected_names != found_names:
            missing = expected_names - found_names
            extra = found_names - expected_names
            details = []
            if missing:
                details.append(f"missing={sorted(missing)!r}")
            if extra:
                details.append(f"extra={sorted(extra)!r}")
            raise ValueError(f"Mismatched optimizer collection state_dict keys: {', '.join(details)}")

        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state_dict[name])
        self._tag_param_groups()

    def _tag_param_groups(self):
        for name, optimizer in self.optimizers.items():
            for param_group in optimizer.param_groups:
                param_group["optimizer_name"] = name

    def _validate_unique_parameters(self):
        seen_parameters: dict[int, tuple[str, str]] = {}
        for optimizer_name, optimizer in self.optimizers.items():
            for param_group in optimizer.param_groups:
                params = param_group["params"]
                param_names = param_group.get("param_names", [""] * len(params))
                for param, param_name in zip(params, param_names, strict=True):
                    parameter_id = id(param)
                    if parameter_id in seen_parameters:
                        previous_optimizer_name, previous_param_name = seen_parameters[parameter_id]
                        raise ValueError(
                            "Parameter is assigned to multiple optimizers: "
                            f"{previous_param_name!r} in {previous_optimizer_name!r} and "
                            f"{param_name!r} in {optimizer_name!r}"
                        )
                    seen_parameters[parameter_id] = (optimizer_name, param_name)


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


def build_optimizer(
    factory_or_factories: OptimizerFactory | Mapping[str, OptimizerFactory],
    named_parameters: Iterable[tuple[str, nn.Parameter]],
) -> Optimizer | OptimizerCollection:
    """Builds either a single optimizer or a named optimizer collection."""
    named_parameters = tuple(named_parameters)
    if isinstance(factory_or_factories, OptimizerFactory):
        return factory_or_factories(named_parameters)

    optimizers = {}
    for name, factory in factory_or_factories.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Optimizer names must be non-empty strings")
        optimizers[name] = factory(named_parameters)
    return OptimizerCollection(optimizers)
