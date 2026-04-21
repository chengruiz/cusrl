import pytest
import torch

from cusrl.nn.layer.rms import RunningMeanStd


def test_running_mean_std_applies_groups_and_excluded_indices():
    normalizer = RunningMeanStd(
        num_channels=4,
        groups=[slice(0, 2)],
        excluded_indices=slice(3, 4),
        clamp=None,
    )
    batch_mean = torch.tensor([1.0, 3.0, 5.0, 7.0])
    batch_var = torch.tensor([4.0, 16.0, 25.0, 49.0])

    normalizer.update_from_stats(batch_mean, batch_var, batch_count=2, synchronize=False)

    expected_mean = torch.tensor([2.0, 2.0, 5.0, 0.0])
    expected_var = torch.tensor([11.0, 11.0, 25.0, 1.0])
    assert torch.allclose(normalizer.mean, expected_mean)
    assert torch.allclose(normalizer.var, expected_var)
    assert torch.allclose(normalizer.std, torch.sqrt(expected_var + normalizer.epsilon))
    assert normalizer.count == 2


def test_running_mean_std_normalize_inplace_respects_clamp():
    normalizer = RunningMeanStd(num_channels=2, clamp=1.0)
    normalizer.mean.copy_(torch.tensor([0.0, 1.0]))
    normalizer.var.copy_(torch.tensor([1.0, 4.0]))
    normalizer.std.copy_(torch.sqrt(normalizer.var + normalizer.epsilon))
    input = torch.tensor([[3.0, -5.0]])

    output = normalizer.normalize_(input.clone())

    assert torch.allclose(output, torch.tensor([[1.0, -1.0]]))


def test_running_mean_std_clear_and_extra_state_roundtrip():
    normalizer = RunningMeanStd(num_channels=2, clamp=None)
    normalizer.update(torch.tensor([[1.0, 3.0], [3.0, 5.0]]), synchronize=False)
    state = normalizer.get_extra_state()

    restored = RunningMeanStd(num_channels=2, clamp=None)
    restored.mean.copy_(normalizer.mean)
    restored.var.copy_(normalizer.var)
    restored.std.copy_(normalizer.std)
    restored.set_extra_state(state)

    assert restored.count == normalizer.count

    normalizer.clear()
    assert torch.allclose(normalizer.mean, torch.zeros(2))
    assert torch.allclose(normalizer.var, torch.ones(2))
    assert torch.allclose(normalizer.std, torch.ones(2))
    assert normalizer.count == 0


def test_running_mean_std_rejects_invalid_extra_state():
    normalizer = RunningMeanStd(num_channels=1)

    with pytest.raises(ValueError, match="must be non-negative"):
        normalizer.set_extra_state(torch.tensor(-1))


def test_running_mean_std_rejects_overlap_between_groups_and_excluded_indices():
    with pytest.raises(ValueError, match="must not overlap"):
        RunningMeanStd(
            num_channels=4,
            groups=[slice(0, 2)],
            excluded_indices=[0],
        )
