from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import numpy as np

from cusrl.template.player import Player, PlayerHook
from cusrl.utils.misc import to_numpy
from cusrl.utils.typing import Array

__all__ = ["SaveTransition"]


class SaveTransition(PlayerHook):
    """Saves transition data collected during playing to .npz files.

    This hook accumulates transition dictionary items (e.g., observations,
    actions, rewards) in a buffer and periodically writes them to disk using
    `numpy.savez`.

    Args:
        output_path (str | None, optional):
            The target file path. If ``None``, a timestamped filename is
            generated. Defaults to ``None``.
        keys (Iterable[str], optional):
            A list of keys to extract from the transition dictionary. Defaults
            to ``("observation", "reward", "terminated", "truncated",
            "action")``.
        save_interval (int | None, optional):
            The number of steps between file flushes. If provided, output files
            will be sharded. If ``None``, all data is saved to a single file
            upon closing. Defaults to ``None``.
    """

    DEFAULT_KEYS = ("observation", "reward", "terminated", "truncated", "action")

    def __init__(
        self,
        output_path: str | None = None,
        keys: Iterable[str] = DEFAULT_KEYS,
        save_interval: int | None = None,
    ):
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"transition_{timestamp}.npz"
        elif not output_path.endswith(".npz"):
            output_path += ".npz"
        self.output_path: str = output_path
        self.keys: tuple[str, ...] = tuple(keys)
        self.save_interval: int | None = save_interval

        self.shard_index: int = 0
        self.buffer = defaultdict(list)

    def init(self, player: Player):
        super().init(player)
        if self.save_interval is not None and self.save_interval <= 0:
            raise ValueError("'save_interval' must be a positive integer or None.")

        self.shard_index = 0
        self.buffer.clear()

    def step(self, step: int, transition: dict[str, Array], metrics: dict[str, float]):
        for key in self.keys:
            self.buffer[key].append(to_numpy(transition[key]))
        if self.save_interval is not None and self.player.step % self.save_interval == 0:
            self.flush()

    def close(self):
        self.flush()

    def flush(self):
        if self.player.step == 0:
            return

        arrays = {key: np.stack(value, axis=0) for key, value in self.buffer.items()}
        output_path = Path(self.output_path)
        if self.save_interval is not None:
            output_path = output_path.with_name(f"{output_path.stem}_{self.shard_index:06d}.npz")
        np.savez(output_path, **arrays)
        self.shard_index += 1
        self.buffer.clear()
