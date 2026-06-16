from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

__all__ = ["OptimizerCollection", "OptimizerFactory", "build_optimizer"]

OptimizerParamSelector = Callable[[str, nn.Parameter], bool]
OptimizerGroupOverride = tuple[str | OptimizerParamSelector, dict[str, Any]]


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
    """Builds a PyTorch optimizer with configurable parameter grouping.

    ``defaults`` defines the base optimizer kwargs. They are passed to the
    optimizer constructor and also used for parameters that do not match any
    configured override rule.

    ``group_overrides`` can provide prefix or callable selector rules such as
    ``("actor", {"lr": 1e-4})`` or
    ``(lambda name, p: p.ndim == 2, {"lr": 1e-4})``. Rules are checked in
    order, and the first matching rule supplies the group kwargs. Parameters
    with ``requires_grad=False`` are skipped. ``param_filter`` can be used to
    keep only parameters accepted by a prefix, prefix sequence, or callable.

    Args:
        cls (str | type[Optimizer]):
            Optimizer class (for example, ``torch.optim.Adam``) or the name of
            a class exposed from ``torch.optim``.
        defaults (dict[str, Any] | None, optional):
            Base optimizer keyword arguments. These are passed to the optimizer
            constructor and reused for unmatched parameters.
        group_overrides (Sequence[OptimizerGroupOverride] | None, optional):
            Ordered selector rules for optimizer keyword overrides. Rules are
            checked before falling back to ``defaults``.
        param_filter (str | Sequence[str] | OptimizerParamSelector | None, optional):
            Optional filter used to keep only a subset of the parameters before
            grouping. If ``None``, all parameters are kept.
    """

    def __init__(
        self,
        cls: str | type[Optimizer],
        defaults: dict[str, Any] | None = None,
        group_overrides: Sequence[OptimizerGroupOverride] | None = None,
        param_filter: str | Sequence[str] | OptimizerParamSelector | None = None,
    ):
        self.cls = cls
        self.defaults = defaults or {}
        self.group_overrides = tuple(group_overrides or ())

        for rule in self.group_overrides:
            try:
                selector, options = rule
            except (TypeError, ValueError) as error:
                raise TypeError("Optimizer group overrides must be (selector, options) tuples") from error
            if isinstance(selector, str):
                if not selector:
                    raise ValueError("Empty prefixes are not allowed in optimizer group overrides")
            elif not callable(selector):
                raise TypeError("Group override selector must be a parameter-name prefix string or a callable")
            if not isinstance(options, dict):
                raise TypeError("Group override options must be a dict")

        if isinstance(param_filter, str):
            if not param_filter:
                raise ValueError("Empty prefixes are not allowed in 'param_filter'")
            self.param_filter = param_filter
        elif param_filter is None or callable(param_filter):
            self.param_filter = param_filter
        else:
            self.param_filter = tuple(param_filter)
            for prefix in self.param_filter:
                if not isinstance(prefix, str):
                    raise TypeError("'param_filter' prefixes must be strings")
                if not prefix:
                    raise ValueError("Empty prefixes are not allowed in 'param_filter'")

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
            if not self._is_param_selected(name, param):
                continue
            rule_index, group_options = self._get_matched_group(name, param)
            param_group = param_groups.get(rule_index)
            if param_group is None:
                param_group = {"param_names": [], "params": [], **group_options}
                param_groups[rule_index] = param_group
            param_group["param_names"].append(name)
            param_group["params"].append(param)
        if not param_groups:
            raise ValueError("No trainable parameters matched the optimizer filter")
        return optim_cls(param_groups.values(), **self.defaults)

    def _is_param_selected(self, name: str, parameter: nn.Parameter) -> bool:
        if self.param_filter is None:
            return True
        if isinstance(self.param_filter, str):
            return name == self.param_filter or name.startswith(f"{self.param_filter}.")
        if callable(self.param_filter):
            return bool(self.param_filter(name, parameter))
        return any(name == prefix or name.startswith(f"{prefix}.") for prefix in self.param_filter)

    def _get_matched_group(self, name: str, parameter: nn.Parameter) -> tuple[int, dict[str, Any]]:
        for index, (selector, options) in enumerate(self.group_overrides):
            if isinstance(selector, str):
                matched = name == selector or name.startswith(f"{selector}.")
            else:
                matched = selector(name, parameter)
            if matched:
                return index, options
        return -1, self.defaults


def build_optimizer(
    factory_or_factories: OptimizerFactory | Mapping[str, OptimizerFactory],
    named_parameters: Iterable[tuple[str, nn.Parameter]],
) -> Optimizer | OptimizerCollection:
    """Builds either a single optimizer or a named optimizer collection."""
    named_parameters = tuple(named_parameters)
    remaining_named_parameters = tuple((name, param) for name, param in named_parameters if param.requires_grad)

    def remove_optimized_parameters(optimizer: Optimizer):
        optimized_parameter_ids = {id(param) for group in optimizer.param_groups for param in group["params"]}
        return tuple(
            (param_name, param)
            for param_name, param in remaining_named_parameters
            if id(param) not in optimized_parameter_ids
        )

    if isinstance(factory_or_factories, OptimizerFactory):
        optimizer = factory_or_factories(remaining_named_parameters)
        remaining_named_parameters = remove_optimized_parameters(optimizer)
        if remaining_named_parameters:
            names = [name for name, _ in remaining_named_parameters]
            raise ValueError(f"Trainable parameters were not assigned to any optimizer: {names!r}")
        return optimizer

    optimizers = {}
    for name, factory in factory_or_factories.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Optimizer names must be non-empty strings")
        optimizer = factory(remaining_named_parameters)
        remaining_named_parameters = remove_optimized_parameters(optimizer)
        optimizers[name] = optimizer
    if remaining_named_parameters:
        names = [name for name, _ in remaining_named_parameters]
        raise ValueError(f"Trainable parameters were not assigned to any optimizer: {names!r}")
    return OptimizerCollection(optimizers)
