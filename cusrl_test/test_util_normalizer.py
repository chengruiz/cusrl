import numpy as np
import pytest
import torch

from cusrl.utils.normalizer import (
    RunningMeanStd,
    mean_var_count,
    merge_mean_var_,
)


def test_mean_var_count_numpy():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    mean, var, count = mean_var_count(arr, uncentered=False)
    assert count == 2
    np.testing.assert_allclose(mean, np.array([2.0, 3.0]))
    np.testing.assert_allclose(var, np.array([1.0, 1.0]))


def test_mean_var_count_numpy_uncentered():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    mean, var, count = mean_var_count(arr, uncentered=True)
    assert count == 2
    np.testing.assert_allclose(mean, np.zeros(2))
    # uncentered var = mean of squares
    np.testing.assert_allclose(var, np.array([(1.0**2 + 3.0**2) / 2, (2.0**2 + 4.0**2) / 2]))


def test_mean_var_count_empty_torch():
    x = torch.empty((0, 3))
    mean, var, count = mean_var_count(x, uncentered=False)
    assert count == 0
    assert torch.allclose(mean, torch.zeros(3))
    assert torch.allclose(var, torch.ones(3))


def test_mean_var_count_torch():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mean, var, count = mean_var_count(x, uncentered=False)
    assert count == 2
    assert torch.allclose(mean, torch.tensor([2.0, 3.0]))
    assert torch.allclose(var, torch.tensor([1.0, 1.0]))


def test_merge_mean_var():
    old_mean = torch.tensor([1.0])
    old_var = torch.tensor([1.0])
    merge_mean_var_(old_mean, old_var, w_old=1.0, new_mean=torch.tensor([3.0]), new_var=torch.tensor([4.0]), w_new=1.0)
    # merged mean = (1 + 3) / 2 = 2
    assert torch.allclose(old_mean, torch.tensor([2.0]))
    # merged var = .5*1 + .5*4 + (1-3)^2*(.5*.5) = .5 + 2 + 1 = 3.5
    assert pytest.approx(old_var.item(), rel=1e-6) == 3.5


def test_running_mean_std_basic():
    rms = RunningMeanStd(num_channels=2, clamp=None)
    data = torch.randn(10, 2, 2).clamp(-5, 5)
    for i in range(10):
        rms.update(data[i])
    assert torch.allclose(rms.mean, data.mean(dim=(0, 1)), atol=1e-6)
    assert torch.allclose(rms.var, data.var(dim=(0, 1), unbiased=False), atol=1e-6)
    assert torch.allclose(rms.std, torch.sqrt(rms.var + rms.epsilon), atol=1e-6)
    # normalize then unnormalize
    normed = rms.normalize(data)
    unnormed = rms.unnormalize(normed)
    assert torch.allclose(unnormed, data, atol=1e-6)
