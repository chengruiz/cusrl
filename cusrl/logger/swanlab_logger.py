import os
from typing import Literal

import cusrl
from cusrl.template.logger import Logger, LoggerFactory

__all__ = ["Swanlab"]


class SwanlabFactory(LoggerFactory):
    def __init__(
        self,
        log_dir: str | os.PathLike,
        name: str | None = "",
        interval: int = 1,
        add_datetime_prefix: bool = True,
        project: str | None = None,
        workspace: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        config: dict | str | None = None,
        mode: Literal["disabled", "cloud", "local", "offline", None] = None,
        load: str | None = None,
        public: bool | None = None,
        id: str | None = None,
        resume: Literal["must", "allow", "never"] | None = None,
        **kwargs,
    ):
        super().__init__(
            log_dir=log_dir,
            name=name,
            interval=interval,
            add_datetime_prefix=add_datetime_prefix,
        )
        self.project = project
        self.workspace = workspace
        self.description = description
        self.tags = tags
        self.config = config
        self.mode = mode
        self.load = load
        self.public = public
        self.id = id
        self.resume = resume
        self.kwargs = kwargs

    def __call__(self):
        return Swanlab(
            log_dir=self.log_dir,
            name=self.name,
            interval=self.interval,
            add_datetime_prefix=self.add_datetime_prefix,
            project=self.project,
            workspace=self.workspace,
            description=self.description,
            tags=self.tags,
            config=self.config,
            mode=self.mode,
            load=self.load,
            public=self.public,
            id=self.id,
            resume=self.resume,
            **self.kwargs,
        )


class Swanlab(Logger):
    Factory = SwanlabFactory

    def __init__(
        self,
        log_dir: str | os.PathLike,
        name: str | None = "",
        interval: int = 1,
        add_datetime_prefix: bool = True,
        **kwargs,
    ):
        try:
            import swanlab
        except ImportError:
            raise ImportError("Install SwanLab with 'pip install swanlab' to use the SwanLab logger")
        if cusrl.utils.is_main_process():
            self.swanlab_run = swanlab.init(experiment_name=name, **kwargs)
            self.provider = swanlab
        else:
            self.swanlab_run = self.provider = None

        super().__init__(
            log_dir=log_dir,
            name=name,
            interval=interval,
            add_datetime_prefix=add_datetime_prefix,
        )

    def _log_impl(self, data: dict[str, float], iteration: int):
        if self.provider is None:
            return
        self.provider.log(data, step=iteration)
