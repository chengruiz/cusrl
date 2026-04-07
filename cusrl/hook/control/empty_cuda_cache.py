import torch

from cusrl.template import Hook

__all__ = ["EmptyCudaCache"]


class EmptyCudaCache(Hook):
    """A hook that empties the CUDA cache after each update."""

    def post_update(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
