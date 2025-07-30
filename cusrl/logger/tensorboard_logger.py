from cusrl.template.logger import Logger

__all__ = ["Tensorboard"]


class TensorboardFactory:
    def __init__(
        self,
        log_dir: str,
        name: str | None = None,
        interval: int = 1,
        add_datetime_prefix: bool = True,
        **kwargs,
    ):
        self.log_dir = log_dir
        self.name = name
        self.interval = interval
        self.add_datetime_prefix = add_datetime_prefix
        self.kwargs = kwargs

    def __call__(self):
        return Tensorboard(
            log_dir=self.log_dir,
            name=self.name,
            interval=self.interval,
            add_datetime_prefix=self.add_datetime_prefix,
            **self.kwargs,
        )


class Tensorboard(Logger):
    Factory = TensorboardFactory

    def __init__(
        self,
        log_dir: str,
        name: str | None = None,
        interval: int = 1,
        add_datetime_prefix: bool = True,
        **kwargs,
    ):
        from torch.utils.tensorboard import SummaryWriter

        super().__init__(
            log_dir=log_dir,
            name=name,
            interval=interval,
            add_datetime_prefix=add_datetime_prefix,
        )
        self.provider = SummaryWriter(log_dir=self.log_dir, **kwargs)

    def _log_impl(self, data: dict[str, float], iteration: int):
        for key, val in data.items():
            self.provider.add_scalar(key, val, global_step=iteration)
