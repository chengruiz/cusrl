from torch import nn

from cusrl.template import Hook

__all__ = ["GradientClipping"]


class GradientClipping(Hook):
    """Clips parameter gradients before the optimizer step.

    Parameters are assigned to the longest matching prefix in ``groups``. Any
    parameter name that does not match a configured prefix falls back to the
    default ``max_grad_norm`` limit. Prefixes match either an exact parameter
    name or a module prefix followed by ``"."``.

    Args:
        max_grad_norm (float | None, optional):
            Maximum norm for parameters outside any named group. If ``None``,
            clipping is disabled for the default group. Defaults to ``1.0``.
        groups (dict[str, float | None] | None, optional):
            Mapping from parameter-name prefixes to per-group clipping limits.
            Longer prefixes take precedence over shorter ones. Defaults to
            ``None``.
        **kwargs (float | None):
            Additional per-prefix clipping limits. Keyword arguments are merged
            into ``groups`` and override duplicate prefixes.

    Raises:
        ValueError:
            If ``max_grad_norm`` or any group limit is negative, or if a group
            prefix is empty.
    """

    def __init__(
        self,
        max_grad_norm: float | None = 1.0,
        groups: dict[str, float | None] | None = None,
        **kwargs: float | None,
    ):
        super().__init__(training_only=True)

        if max_grad_norm is not None and max_grad_norm < 0:
            raise ValueError("'max_grad_norm' must be non-negative")

        self.max_grad_norm = max_grad_norm
        groups = (groups or {}) | kwargs
        for prefix, group_max_grad_norm in groups.items():
            if not prefix:
                raise ValueError("Empty prefixes are not allowed; use 'max_grad_norm' for the default group")
            if group_max_grad_norm is not None and group_max_grad_norm < 0:
                raise ValueError(f"'max_grad_norm' for prefix '{prefix}' must be non-negative")

        # Sort by length of prefix (longest first for more specific matching)
        self.groups = dict(sorted(groups.items(), key=lambda x: len(x[0]), reverse=True))

    def pre_optim(self, optimizer):
        """Clips each gradient group and records its pre-clip norm.

        The optimizer is expected to expose ``param_names`` in each parameter
        group. Parameters without a matching prefix are treated as part of the
        default group.
        """

        prefixed_parameters = {"": [], **{prefix: [] for prefix in self.groups}}
        for param_group in optimizer.param_groups:
            params = param_group["params"]
            param_names = param_group.get("param_names", [""] * len(params))
            for param, name in zip(params, param_names, strict=True):
                prefix = self._match_prefix(name)
                prefixed_parameters[prefix].append(param)
        # Clip gradients for each group
        for prefix, params in prefixed_parameters.items():
            if params and (max_grad_norm := self.groups.get(prefix, self.max_grad_norm)) is not None:
                grad_norm = nn.utils.clip_grad_norm_(params, max_grad_norm)
                self.agent.record(**{f"grad_norm/{prefix or 'default'}": grad_norm})

    def _match_prefix(self, name: str) -> str:
        """Returns the most specific configured prefix that matches ``name``."""

        for prefix in self.groups:
            if name == prefix or name.startswith(f"{prefix}."):
                return prefix
        return ""
