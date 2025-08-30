import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TypeAlias

import torch

__all__ = [
    "LoggerFactory",
    "LoggerFactoryLike",
    "Logger",
    "make_logger_factory",
]


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
    """A base class for logging experiment data.

    This class handles the creation of a structured log directory, saving model
    checkpoints, and logging metric data. It is designed to be subclassed to
    implement specific logging backends (e.g., TensorBoard, Weights & Biases)
    by overriding the `_log_impl` method.

    The logger creates a directory structure as follows:
    `[log_dir]/`
        `[timestamp]:[name]/`
            `info/` - For storing text-based information.
            `ckpt/` - For storing model checkpoints.
        `latest` -> symlink to `[timestamp]:[name]/`

    Args:
        log_dir (str):
            The root directory where logs will be stored.
        name (str | None, optional):
            A specific name for the experiment run. If None, the name is empty.
            Defaults to None.
        interval (int, optional):
            The interval at which to log data. If greater than 1, data is
            averaged over the interval before logging. Defaults to 1.
        add_datetime_prefix (bool, optional):
            If True, a timestamp prefix (YYYY-MM-DD-HH-MM-SS) is added to the
            experiment directory name. Defaults to True.
    """

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


def make_logger_factory(
    logger_type: str | None = None,
    log_dir: str | None = None,
    name: str | None = None,
    interval: int = 1,
    add_datetime_prefix: bool = True,
    **kwargs,
) -> LoggerFactoryLike | None:
    if log_dir is None:
        return None
    if logger_type is None or logger_type.lower() == "none":
        return Logger.Factory(log_dir=log_dir, name=name, interval=interval, **kwargs)
    logger_cls_dict = {cls.__name__.lower(): cls for cls in Logger.__subclasses__()}
    logger_cls = logger_cls_dict[logger_type.lower()]
    return logger_cls.Factory(
        log_dir=log_dir,
        name=name,
        interval=interval,
        add_datetime_prefix=add_datetime_prefix,
        **kwargs,
    )
