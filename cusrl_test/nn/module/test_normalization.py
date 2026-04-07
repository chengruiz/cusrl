import numpy as np
import pytest
import torch

import cusrl
from cusrl.nn.module.normalization import Denormalization


def test_normalization_and_denormalization_are_inverses():
    mean = torch.tensor([1.0, -1.0])
    std = torch.tensor([2.0, 4.0])
    normalization = cusrl.Normalization(mean, std)
    denormalization = Denormalization(mean, std)
    input = torch.tensor([[3.0, 7.0], [1.0, -1.0]])

    normalized = normalization(input)
    restored = denormalization(normalized)

    assert torch.allclose(restored, input)
    assert not normalization.mean.requires_grad
    assert not normalization.std.requires_grad


def test_normalization_factories_accept_array_like_inputs():
    normalization = cusrl.Normalization.Factory(mean=[1.0, -1.0], std=np.array([2.0, 4.0]))()
    denormalization = Denormalization.Factory(mean=np.array([1.0, -1.0]), std=[2.0, 4.0])()
    input = torch.tensor([[3.0, 7.0]])

    restored = denormalization(normalization(input))
    assert torch.allclose(restored.float(), input)


def test_normalization_factories_validate_requested_dimensions():
    norm_factory = cusrl.Normalization.Factory(mean=[0.0, 1.0], std=[1.0, 2.0])
    denorm_factory = Denormalization.Factory(mean=[0.0, 1.0], std=[1.0, 2.0])

    with pytest.raises(ValueError, match="Input dimension mismatch"):
        norm_factory(input_dim=3)
    with pytest.raises(ValueError, match="Output dimension mismatch"):
        denorm_factory(output_dim=3)
