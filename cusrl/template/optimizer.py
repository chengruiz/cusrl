from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

__all__ = ["OptimizerFactory"]


class OptimizerFactory:
    """A factory for creating PyTorch optimizers with parameter-specific settings.

    This class allows for the flexible configuration of an optimizer where different
    groups of parameters can have different hyperparameters (e.g., learning rate).
    Parameter groups are defined by their name prefixes.

    The factory is configured with an optimizer class, default hyperparameters, and
    any number of parameter groups specified by a prefix and their corresponding
    hyperparameters. When called with a model's named parameters, it groups them
    according to the longest matching prefix and constructs the optimizer.

    Args:
        cls (str | type[Optimizer]):
            The optimizer class (e.g., `torch.optim.Adam`) or its name as a
            string (e.g., "Adam").
        defaults (dict[str, Any] | None, optional):
            A dictionary of default hyperparameters for the optimizer. These are
            used for any parameter that doesn't match a specific group.
        **optim_groups (dict[str, Any]):
            Keyword arguments where each key is a parameter name prefix and the
            value is a dictionary of hyperparameters for parameters matching that prefix.
    """

    def __init__(
        self,
        cls: str | type[Optimizer],
        defaults: dict[str, Any] | None = None,
        **optim_groups: dict[str, Any],
    ):
        self.cls: type[Optimizer] = getattr(torch.optim, cls) if isinstance(cls, str) else cls
        self.defaults = defaults or {}

        optim_groups[""] = self.defaults
        # Sort by length of prefix
        self.optim_groups = dict(sorted(optim_groups.items(), key=lambda x: len(x[0]), reverse=True))

    def __call__(self, named_parameters: Iterable[tuple[str, nn.Parameter]]) -> Optimizer:
        param_groups = {}
        for name, param in named_parameters:
            if not param.requires_grad:
                continue
            prefix = self._match_prefix(name)
            param_group = param_groups.get(prefix, None)
            if param_group is None:
                param_group = {"param_names": [], "params": [], **self.optim_groups[prefix]}
                param_groups[prefix] = param_group
            param_group["param_names"].append(name)
            param_group["params"].append(param)
        return self.cls(param_groups.values(), **self.defaults)

    def _match_prefix(self, name):
        matched_prefix = ""
        for prefix in self.optim_groups:
            if name == prefix or name.startswith(f"{prefix}.") or not prefix:
                matched_prefix = prefix
                break
        return matched_prefix
