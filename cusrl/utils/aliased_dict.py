from typing import Generic, TypeVar, overload

__all__ = ["AliasedDict"]

_K = TypeVar("_K")
_V = TypeVar("_V")
_D = TypeVar("_D")


class AliasedDict(Generic[_K, _V], dict[_K, _V]):
    """A dictionary-like class that supports key aliasing.

    This class extends the standard :class:`dict` to allow one or more alias
    keys to refer to a single canonical key. All standard dictionary methods
    that operate on keys (:func:`__getitem__`, :func:`get`, :func:`__setitem__`,
    etc.) are overridden to resolve aliases before performing the operation.
    """

    __slots__ = ("_aliases",)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._aliases: dict[_K, _K] = {}

    def register_alias(self, alias: _K, key: _K):
        """Register an alias for an existing (or future) key.

        If the key does not yet exist in the mapping, the alias is still stored
        and will resolve once the key is set.
        """
        if alias == key:
            return
        self._aliases[alias] = key

    def canonical_key(self, name: _K) -> _K:
        """Return the canonical key given a name or alias."""
        return self._aliases.get(name, name)

    # Access overrides
    def __getitem__(self, key: _K) -> _V:
        return super().__getitem__(self.canonical_key(key))

    @overload
    def get(self, key: _K, default: None = None, /) -> _V | None: ...
    @overload
    def get(self, key: _K, default: _V, /) -> _V: ...
    @overload
    def get(self, key: _K, default: _D, /) -> _V | _D: ...
    def get(self, key: _K, default: _D | None = None) -> _V | _D | None:
        return super().get(self.canonical_key(key), default)

    def __setitem__(self, key: _K, value: _V):
        super().__setitem__(self.canonical_key(key), value)

    def clear(self):
        super().clear()
        self._aliases.clear()

    # Utilities
    def copy(self):
        new = AliasedDict(super().copy())
        new._aliases = self._aliases.copy()
        return new

    def __repr__(self):
        return f"AliasedDict({super().__repr__()}, aliases={self._aliases})"
