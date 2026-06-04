import pytest
import torch

import cusrl
from cusrl.hook.on_policy.ppo import _ppo_surrogate_loss


def test_ppo_surrogate_loss_matches_clipped_reference_values():
    advantage = torch.tensor([[1.0], [-2.0]])
    prob_ratio = torch.tensor([[1.5], [0.5]])

    loss = _ppo_surrogate_loss(advantage, prob_ratio, clip_ratio=0.2)

    assert loss.item() == pytest.approx(0.2)


def test_ppo_surrogate_loss_hook_validates_scalar_advantage_dimension():
    hook = cusrl.hook.PpoSurrogateLoss(clip_ratio=0.2, weight=2.0)
    batch = {
        "advantage": torch.tensor([[1.0, 2.0]]),
        "action_prob_ratio": torch.ones(1, 2),
    }

    with pytest.raises(ValueError, match="Expected advantage"):
        hook.objective({}, batch)


def test_entropy_loss_returns_negative_weighted_entropy_mean():
    hook = cusrl.hook.EntropyLoss(weight=0.5)
    loss = hook.objective({}, {"curr_entropy": torch.tensor([[1.0], [3.0]])})

    assert loss["entropy_loss"].item() == pytest.approx(-1.0)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: cusrl.hook.PpoSurrogateLoss(clip_ratio=0.0),
        lambda: cusrl.hook.PpoSurrogateLoss(weight=-1.0),
        lambda: cusrl.hook.EntropyLoss(weight=-1.0),
    ],
)
def test_ppo_loss_hooks_validate_positive_configuration(factory):
    with pytest.raises(ValueError):
        factory()
