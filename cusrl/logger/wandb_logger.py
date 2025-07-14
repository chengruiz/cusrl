from typing import Literal

from cusrl.template.logger import Logger

__all__ = ["Wandb"]


class WandbFactory:
    def __init__(
        self,
        log_dir: str,
        name: str | None = None,
        interval: int = 1,
        project: str | None = None,
        dir: str | None = None,
        id: str | None = None,
        mode: Literal["online", "offline", "disabled"] | None = None,
        resume: bool | Literal["allow", "never", "must", "auto"] | None = None,
        save_code: bool | None = None,
        tensorboard: bool | None = None,
        sync_tensorboard: bool | None = None,
        **kwargs,
    ):
        self.log_dir = log_dir
        self.name = name
        self.interval = interval
        self.project = project
        self.dir = dir
        self.id = id
        self.mode = mode
        self.resume = resume
        self.save_code = save_code
        self.tensorboard = tensorboard
        self.sync_tensorboard = sync_tensorboard
        self.kwargs = kwargs

    def __call__(self):
        return Wandb(
            log_dir=self.log_dir,
            name=self.name,
            interval=self.interval,
            project=self.project,
            dir=self.dir,
            id=self.id,
            mode=self.mode,
            resume=self.resume,
            save_code=self.save_code,
            tensorboard=self.tensorboard,
            sync_tensorboard=self.sync_tensorboard,
            **self.kwargs,
        )


class Wandb(Logger):
    Factory = WandbFactory

    def __init__(self, log_dir: str, name: str | None = None, interval: int = 1, **kwargs):
        super().__init__(log_dir, name, interval)
        try:
            import wandb
        except ImportError:
            raise ImportError("Please run 'pip install wandb' to use wandb logger.")

        self.run = wandb.init(name=name, **kwargs)
        self.provider = wandb

    def _log_impl(self, data: dict[str, float], iteration: int):
        self.provider.log(data, step=iteration)
