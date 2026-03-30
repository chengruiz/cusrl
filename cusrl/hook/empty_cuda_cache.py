from dataclasses import dataclass

import torch

from cusrl.template import Hook, HookFactory

__all__ = ["EmptyCudaCache"]


class EmptyCudaCache(Hook):
    """A hook that empties the CUDA cache after each update."""

    @dataclass
    class Factory(HookFactory["EmptyCudaCache"]):
        @classmethod
        def get_hook_type(cls):
            return EmptyCudaCache

    def post_update(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
