import torch
from torch import nn

from cusrl.preset.optimizer import AdamFactory, AdamWFactory


def test_adam_factory_builds_torch_adam_with_requested_defaults():
    module = nn.Linear(2, 1)

    optimizer = AdamFactory(defaults={"lr": 0.003})(module.named_parameters())

    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]["lr"] == 0.003


def test_adamw_factory_builds_torch_adamw_with_requested_defaults():
    module = nn.Linear(2, 1)

    optimizer = AdamWFactory(defaults={"lr": 0.004, "weight_decay": 0.1})(module.named_parameters())

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 0.004
    assert optimizer.param_groups[0]["weight_decay"] == 0.1
