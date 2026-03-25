import cusrl
from cusrl.module import ExponentialBijector, IdentityBijector, SigmoidBijector, SoftplusBijector, make_bijector


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
