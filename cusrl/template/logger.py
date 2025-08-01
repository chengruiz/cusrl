import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TypeAlias

import torch

__all__ = ["LoggerFactory", "LoggerFactoryLike", "Logger"]


@dataclass(slots=True)
class LoggerFactory:
    log_dir: str
    name: str | None = None
    interval: int = 1
    add_datetime_prefix: bool = True

    def __call__(self):
        return Logger(
            log_dir=self.log_dir,
            name=self.name,
            interval=self.interval,
            add_datetime_prefix=self.add_datetime_prefix,
        )


LoggerFactoryLike: TypeAlias = Callable[[], "Logger"]


class Logger:
    Factory = LoggerFactory

    def __init__(
        self,
        log_dir: str,
        name: str | None = None,
        interval: int = 1,
        add_datetime_prefix: bool = True,
    ):
        self.name = name or ""
        if "/" in self.name or "\\" in self.name:
            raise ValueError("'name' should not contain '/' or '\\' characters.")
        if add_datetime_prefix:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            self.name = f"{timestamp}:{self.name}" if self.name else timestamp

        self.log_dir = Path(os.path.join(log_dir, self.name)).absolute()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        symlink_path = self.log_dir / ".." / "latest"
        symlink_path.unlink(missing_ok=True)
        symlink_path.symlink_to(self.log_dir.name, target_is_directory=True)
        self.info_dir = self.log_dir / "info"
        self.info_dir.mkdir(exist_ok=True)
        self.ckpt_dir = self.log_dir / "ckpt"
        self.ckpt_dir.mkdir(exist_ok=True)

        self.interval = interval
        self.data_list = []

    def log(self, data: dict[str, float], iteration: int):
        if self.interval > 1:
            if iteration % self.interval != 0:
                self.data_list.append(data)
            else:
                data = self._collect_data()
                self.data_list.clear()

        self._log_impl(data, iteration)

    def save_checkpoint(self, state_dict, iteration: int):
        torch.save(state_dict, self.ckpt_dir / f"ckpt_{iteration}.pt")

    def save_info(self, info_str: str, filename: str):
        with open(self.info_dir / filename, "w") as f:
            f.write(info_str)

    def _collect_data(self):
        collection = {}
        for data in self.data_list:
            for key, val in data.items():
                if key not in collection:
                    collection[key] = []
                collection[key].append(val)
        return {key: sum(val) / len(val) for key, val in collection.items()}

    def _log_impl(self, data: dict[str, float], iteration: int):
        pass
