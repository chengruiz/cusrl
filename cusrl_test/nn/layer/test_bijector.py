import pytest
import torch

import cusrl
from cusrl.nn.layer import ExponentialBijector, IdentityBijector, SigmoidBijector, SoftplusBijector, make_bijector


def test_make_bijector_supports_bare_specs():
    assert isinstance(make_bijector("identity"), IdentityBijector)
    assert isinstance(make_bijector("exp"), ExponentialBijector)
    assert isinstance(make_bijector("exponential"), ExponentialBijector)
    assert isinstance(make_bijector("sigmoid"), SigmoidBijector)
    assert isinstance(make_bijector("softplus"), SoftplusBijector)


def test_make_bijector_supports_parameterized_specs():
    bijector = make_bijector("sigmoid_0_1_0.01")
    assert isinstance(bijector, SigmoidBijector)
    assert bijector.min_value == 0.0
    assert bijector.max_value == 1.0
    assert bijector.eps == 0.01


def test_adaptive_normal_dist_factory_uses_default_bijector_spec():
    dist = cusrl.AdaptiveNormalDist.Factory()(4, 2)
    assert isinstance(dist.bijector, ExponentialBijector)


@pytest.mark.parametrize("scale", [1.0, 0.5, 2.0])
def test_softplus_bijector_inverse_is_consistent_with_forward(scale):
    bijector = SoftplusBijector(scale=scale)
    x = torch.linspace(bijector.min_value, bijector.max_value, 50)
    roundtrip = bijector.forward(bijector.inverse(x))
    assert torch.allclose(roundtrip, x, atol=1e-5)


def test_softplus_bijector_inverse_large_input_no_overflow():
    bijector = SoftplusBijector(scale=1.0, max_value=200.0)
    x = torch.tensor([100.0, 150.0, 200.0])
    result = bijector.inverse(x)
    assert torch.isfinite(result).all()
    roundtrip = bijector.forward(result)
    assert torch.allclose(roundtrip, x, atol=1e-4)


def test_softplus_bijector_inverse_gradient_finite():
    bijector = SoftplusBijector(scale=1.0, max_value=200.0)
    x = torch.tensor([0.1, 1.0, 50.0, 100.0], requires_grad=True)
    result = bijector.inverse(x)
    result.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_softplus_bijector_float_large_input_no_overflow():
    bijector = SoftplusBijector(scale=1.0, max_value=1000.0)
    inv = bijector.inverse(800.0)
    assert inv == pytest.approx(800.0, abs=1e-4)
    assert bijector.forward(inv) == pytest.approx(800.0, abs=1e-4)


def test_softplus_bijector_inverse_small_float32_input_no_underflow():
    bijector = SoftplusBijector(scale=1.0, min_value=1e-8, max_value=1.0)
    x = torch.tensor([1e-8, 3e-8], dtype=torch.float32)
    result = bijector.inverse(x)
    assert torch.isfinite(result).all()
    roundtrip = bijector.forward(result)
    assert torch.allclose(roundtrip, x, rtol=1e-5, atol=1e-12)


def test_sigmoid_bijector_float_extreme_input_no_overflow():
    bijector = SigmoidBijector(min_value=-2.0, max_value=3.0)
    assert bijector.forward(-1000.0) == pytest.approx(-2.0)
    assert bijector.forward(1000.0) == pytest.approx(3.0)
