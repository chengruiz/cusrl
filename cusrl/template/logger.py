import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TypeAlias

import torch

__all__ = ["LoggerFactory", "LoggerFactoryLike", "Logger"]


@dataclass(slots=True)
class LoggerFactory:
    log_dir: str
    name: str | None = None
    interval: int = 1

    def __call__(self):
        return Logger(self.log_dir, self.name, self.interval)


LoggerFactoryLike: TypeAlias = Callable[[], "Logger"]


class Logger:
    Factory = LoggerFactory

    def __init__(self, log_dir: str, name: str | None = None, interval: int = 1):
        self.name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if name:
            if "/" in name:
                raise ValueError("'name' should not contain '/'.")
            self.name += f":{name}"

        self.log_dir = os.path.abspath(os.path.join(log_dir, self.name))
        os.makedirs(self.log_dir)
        symlink_path = os.path.join(log_dir, "latest")
        if os.path.islink(symlink_path):
            os.remove(symlink_path)
        os.symlink(self.name, symlink_path, target_is_directory=True)
        self.info_dir = os.path.join(self.log_dir, "info")
        os.mkdir(self.info_dir)
        self.ckpt_dir = os.path.join(self.log_dir, "ckpt")
        os.mkdir(self.ckpt_dir)

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
        torch.save(state_dict, os.path.join(self.ckpt_dir, f"ckpt_{iteration}.pt"))

    def save_info(self, info_str: str, filename: str):
        with open(os.path.join(self.info_dir, filename), "w") as f:
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
