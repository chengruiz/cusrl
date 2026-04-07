import pytest
import torch

from cusrl.nn.layer.loss import LOG_SQRT_2PI, GradientPenaltyLoss, L2RegularizationLoss, NormalNllLoss


@pytest.mark.parametrize(
    ("reduction", "expected"),
    [
        ("none", torch.tensor([13.0, 13.0])),
        ("mean", torch.tensor(13.0)),
        ("sum", torch.tensor(26.0)),
    ],
)
def test_gradient_penalty_loss_matches_known_linear_gradient(reduction, expected):
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    weights = torch.tensor([2.0, -3.0])
    output = (input * weights).sum(dim=-1)

    loss = GradientPenaltyLoss(reduction=reduction)(output, input)

    assert torch.allclose(loss, expected)


@pytest.mark.parametrize(
    ("mode", "dist"),
    [
        ("log_var", torch.log(torch.tensor([[0.25, 4.0]]))),
        ("log_std", torch.log(torch.tensor([[0.5, 2.0]]))),
        ("var", torch.tensor([[0.25, 4.0]])),
        ("std", torch.tensor([[0.5, 2.0]])),
    ],
)
def test_normal_nll_loss_accepts_all_variance_parameterizations(mode, dist):
    mean = torch.tensor([[1.0, 2.0]])
    target = torch.tensor([[1.5, 1.0]])
    var = torch.tensor([[0.25, 4.0]])
    expected = 0.5 * (var.log() + (target - mean).square() / var) + LOG_SQRT_2PI

    loss = NormalNllLoss(mode=mode, full=True, reduction="none")((mean, dist), target)

    assert torch.allclose(loss, expected)


def test_normal_nll_loss_supports_concatenated_inputs():
    mean = torch.tensor([[1.0, 2.0]])
    var = torch.tensor([[0.25, 4.0]])
    target = torch.tensor([[1.5, 1.0]])
    input = torch.cat([mean, var], dim=-1)
    expected = 0.5 * (var.log() + (target - mean).square() / var)

    loss = NormalNllLoss(mode="var", reduction="none")(input, target)

    assert torch.allclose(loss, expected)


def test_normal_nll_loss_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="greater than zero"):
        NormalNllLoss(eps=0.0)
    with pytest.raises(ValueError, match="Unsupported mode"):
        NormalNllLoss(mode="precision")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("reduction", "expected"),
    [
        ("none", torch.tensor([[1.0, 4.0], [9.0, 16.0]])),
        ("mean", torch.tensor(7.5)),
        ("sum", torch.tensor(30.0)),
    ],
)
def test_l2_regularization_loss_matches_squared_l2_values(reduction, expected):
    input = torch.tensor([[1.0, -2.0], [3.0, -4.0]])

    loss = L2RegularizationLoss(reduction=reduction)(input)

    assert torch.allclose(loss, expected)
