from types import SimpleNamespace

import pytest
import torch

import cusrl


def test_advantage_reduction_applies_weighted_sum_and_updates_weight_tensor():
    hook = cusrl.hook.AdvantageReduction(reduction="sum", weight=(1.0, 2.0))
    hook.agent = SimpleNamespace(to_tensor=lambda value: torch.as_tensor(value, dtype=torch.float32))
    hook.init()
    batch = {"advantage": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}

    hook.objective({}, batch)
    assert torch.allclose(batch["advantage"], torch.tensor([[5.0], [11.0]]))

    hook.update_attribute("weight", (0.5, 0.5))
    batch = {"advantage": torch.tensor([[2.0, 6.0]])}
    hook.objective({}, batch)
    assert torch.allclose(batch["advantage"], torch.tensor([[4.0]]))


def test_advantage_reduction_supports_mean_and_rejects_unknown_reduction():
    hook = cusrl.hook.AdvantageReduction(reduction="mean")
    hook.agent = SimpleNamespace(to_tensor=lambda value: torch.as_tensor(value, dtype=torch.float32))
    hook.init()
    batch = {"advantage": torch.tensor([[1.0, 3.0]])}

    hook.objective({}, batch)

    assert torch.allclose(batch["advantage"], torch.tensor([[2.0]]))
    with pytest.raises(ValueError, match="Unsupported reduction"):
        cusrl.hook.AdvantageReduction(reduction="max")


def test_advantage_normalization_can_run_on_full_buffer_or_mini_batch():
    advantage = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)

    full_buffer_hook = cusrl.hook.AdvantageNormalization(mini_batch_wise=False, synchronize=False)
    buffer = {"advantage": advantage.clone()}
    full_buffer_hook.pre_update(buffer)
    assert torch.allclose(buffer["advantage"].mean(dim=(0, 1)), torch.zeros(2), atol=1e-6)

    mini_batch_hook = cusrl.hook.AdvantageNormalization(mini_batch_wise=True, synchronize=False)
    batch = {"advantage": advantage.clone()}
    mini_batch_hook.objective({}, batch)
    assert torch.allclose(batch["advantage"].mean(dim=(0, 1)), torch.zeros(2), atol=1e-6)
