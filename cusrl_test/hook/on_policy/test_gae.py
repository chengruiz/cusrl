import pytest
import torch

import cusrl
from cusrl.hook.on_policy.gae import _generalized_advantage_estimation


def test_generalized_advantage_estimation_resets_at_done_boundaries():
    reward = torch.tensor([[[1.0]], [[1.0]], [[1.0]]])
    done = torch.tensor([[[False]], [[True]], [[False]]])
    value = torch.zeros_like(reward)
    next_value = torch.zeros_like(reward)

    advantage = _generalized_advantage_estimation(reward, done, value, next_value, gamma=0.5, lamda=1.0)

    assert torch.allclose(advantage, torch.tensor([[[1.5]], [[1.0]], [[1.0]]]))


def test_gae_hook_writes_advantage_and_separate_value_return():
    hook = cusrl.hook.GeneralizedAdvantageEstimation(gamma=0.5, lamda=1.0, lamda_value=0.0)
    data = {
        "reward": torch.tensor([[[1.0]], [[2.0]]]),
        "done": torch.zeros(2, 1, 1, dtype=torch.bool),
        "value": torch.tensor([[[0.5]], [[1.0]]]),
        "next_value": torch.tensor([[[1.0]], [[0.0]]]),
    }

    hook.pre_update(data)

    assert torch.allclose(data["advantage"], torch.tensor([[[1.5]], [[1.0]]]))
    assert torch.allclose(data["return"], torch.tensor([[[1.5]], [[2.0]]]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gamma": -0.1},
        {"gamma": 1.0},
        {"lamda": -0.1},
        {"lamda": 1.1},
        {"lamda_value": 1.1},
    ],
)
def test_gae_validates_discount_parameters(kwargs):
    with pytest.raises(ValueError):
        cusrl.hook.GeneralizedAdvantageEstimation(**kwargs)
