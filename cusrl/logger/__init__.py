from cusrl.template import make_logger_factory as make_factory  # For compatibility

from .tensorboard_logger import Tensorboard
from .wandb_logger import Wandb

__all__ = ["Tensorboard", "Wandb", "make_factory"]
