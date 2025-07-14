from cusrl.template.logger import Logger, LoggerFactoryLike

__all__ = ["make_factory"]


def make_factory(
    logger_type: str | None = None,
    log_dir: str | None = None,
    name: str | None = None,
    interval: int = 1,
    **kwargs,
) -> LoggerFactoryLike | None:
    if log_dir is None:
        return None
    if logger_type is None or logger_type.lower() == "none":
        return Logger.Factory(log_dir=log_dir, name=name, interval=interval, **kwargs)
    logger_cls_dict = {cls.__name__.lower(): cls for cls in Logger.__subclasses__()}
    return logger_cls_dict[logger_type.lower()].Factory(log_dir=log_dir, name=name, interval=interval, **kwargs)
