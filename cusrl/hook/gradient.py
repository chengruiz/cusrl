from torch import nn

from cusrl.template import Hook

__all__ = ["GradientClipping"]


class GradientClipping(Hook):
    """A hook to clip gradients of model parameters before the optimizer step,
    grouping parameters by name prefixes to handle varying gradient scales
    across modules.

    Args:
        max_grad_norm (float | None, optional):
            Default max norm for gradient clipping. If None, no clipping is
            applied for the default group. Defaults to 1.0.
        **groups (dict[str, float | None]):
            Keyword arguments mapping parameter name prefixes to specific max
            gradient norms.
    """

    def __init__(self, max_grad_norm: float | None = 1.0, **groups: float | None):
        groups[""] = max_grad_norm
        for prefix, max_grad_norm in groups.items():
            if max_grad_norm is not None and max_grad_norm < 0:
                raise ValueError(f"'max_grad_norm' for prefix '{prefix}' must be positive.")
        # Sort by length of prefix (longest first for more specific matching)
        self.groups = dict(sorted(groups.items(), key=lambda x: len(x[0]), reverse=True))

    def pre_optim(self, optimizer):
        prefixed_parameters = {prefix: [] for prefix in self.groups}
        for param_group in optimizer.param_groups:
            params = param_group["params"]
            param_names = param_group.get("param_names", [""] * len(params))
            for param, name in zip(params, param_names, strict=True):
                prefix = self._match_prefix(name)
                prefixed_parameters[prefix].append(param)
        # Clip gradients for each group
        for prefix, params in prefixed_parameters.items():
            if params and (max_grad_norm := self.groups[prefix]) is not None:
                nn.utils.clip_grad_norm_(params, max_grad_norm)

    def _match_prefix(self, name):
        # Find the longest matching prefix (most specific)
        for prefix in self.groups:
            if name == prefix or name.startswith(f"{prefix}."):
                return prefix
        return ""
