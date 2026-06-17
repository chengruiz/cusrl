import pytest
import torch

import cusrl


def test_normal_dist_supports_sampling_metrics_and_deterministic_wrapper():
    torch.manual_seed(0)
    dist = cusrl.NormalDist(input_dim=4, output_dim=2, bijector="exp")
    latent = torch.randn(3, 4)

    dist_params = dist(latent)
    sample, logp = dist.sample_from_dist(dist_params)
    action = dist.deterministic()(latent)

    assert dist_params["mean"].shape == (3, 2)
    assert dist_params["std"].shape == (3, 2)
    assert sample.shape == (3, 2)
    assert logp.shape == (3, 1)
    assert torch.allclose(action, dist_params["mean"])
    assert dist.compute_logp(dist_params, sample).shape == (3, 1)
    assert dist.compute_entropy(dist_params).shape == (3, 1)
    assert torch.allclose(dist.compute_kl_div(dist_params, dist_params), torch.zeros(3, 1), atol=1e-6)


def test_normal_dist_keeps_distribution_math_in_float32():
    dist = cusrl.NormalDist(input_dim=4, output_dim=2, bijector="exp").half()
    latent = torch.randn(3, 4, dtype=torch.float16)

    dist_params = dist(latent)
    sample, logp = dist.sample_from_dist(dist_params)
    half_dist_params = {key: value.half() for key, value in dist_params.items()}

    assert dist_params["mean"].dtype == torch.float32
    assert dist_params["std"].dtype == torch.float32
    assert sample.dtype == torch.float32
    assert logp.dtype == torch.float32
    assert dist.compute_logp(half_dist_params, sample.half()).dtype == torch.float32
    assert dist.compute_entropy(half_dist_params).dtype == torch.float32
    assert dist.compute_kl_div(half_dist_params, half_dist_params).dtype == torch.float32


def test_normal_dist_disables_autocast_for_mean_head():
    torch.manual_seed(0)
    dist = cusrl.NormalDist(input_dim=4, output_dim=2, bijector="exp")
    latent = torch.randn(3, 4)
    expected_mean = dist.mean_head(latent)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        dist_params = dist(latent)

    assert dist_params["mean"].dtype == torch.float32
    assert torch.equal(dist_params["mean"], expected_mean)


def test_adaptive_normal_dist_backward_flag_detaches_std_branch_from_latent():
    torch.manual_seed(0)
    dist = cusrl.AdaptiveNormalDist(input_dim=3, output_dim=2, bijector=None, backward=False)
    latent = torch.randn(4, 3, requires_grad=True)

    expected_grad = torch.autograd.grad(dist.mean_head(latent).sum(), latent, retain_graph=True)[0]
    latent.grad = None

    dist_params = dist(latent)
    (dist_params["mean"].sum() + dist_params["std"].sum()).backward()

    assert torch.allclose(latent.grad, expected_grad)


def test_adaptive_normal_dist_factory_forwards_init_std():
    dist = cusrl.AdaptiveNormalDist.Factory(init_std=0.5, bijector="exp")(3, 2)
    latent = torch.zeros(4, 3)

    dist_params = dist(latent)

    assert torch.allclose(dist_params["std"], torch.full((4, 2), 0.5))


def test_adaptive_normal_dist_keeps_distribution_math_in_float32():
    dist = cusrl.AdaptiveNormalDist(input_dim=4, output_dim=2, bijector="exp").half()
    latent = torch.randn(3, 4, dtype=torch.float16)

    dist_params = dist(latent)
    sample, logp = dist.sample_from_dist(dist_params)

    assert dist_params["mean"].dtype == torch.float32
    assert dist_params["std"].dtype == torch.float32
    assert sample.dtype == torch.float32
    assert logp.dtype == torch.float32


def test_adaptive_normal_dist_disables_autocast_for_distribution_heads():
    torch.manual_seed(0)
    dist = cusrl.AdaptiveNormalDist(input_dim=4, output_dim=2, bijector="exp")
    dist.std_head.weight.data.normal_()
    dist.std_head.bias.data.normal_()
    latent = torch.randn(3, 4)
    expected_mean = dist.mean_head(latent)
    expected_std = dist.bijector(dist.std_head(latent))

    with torch.autocast("cpu", dtype=torch.bfloat16):
        dist_params = dist(latent)

    assert dist_params["mean"].dtype == torch.float32
    assert dist_params["std"].dtype == torch.float32
    assert torch.equal(dist_params["mean"], expected_mean)
    assert torch.equal(dist_params["std"], expected_std)


def test_one_hot_categorical_dist_returns_one_hot_actions_and_statistics():
    dist = cusrl.OneHotCategoricalDist(input_dim=4, output_dim=3)
    dist.mean_head.weight.data.zero_()
    dist.mean_head.bias.data.copy_(torch.tensor([1.0, 3.0, 2.0]))
    latent = torch.zeros(5, 4)

    dist_params = dist(latent)
    mode = dist.determine(latent)
    sample, logp = dist.sample_from_dist(dist_params)

    assert dist_params["logits"].shape == (5, 3)
    assert mode.dtype == dist_params["logits"].dtype
    assert torch.equal(mode, torch.tensor([[0.0, 1.0, 0.0]]).repeat(5, 1))
    assert sample.shape == (5, 3)
    assert torch.equal(sample.sum(dim=-1), torch.ones(5))
    assert logp.shape == (5, 1)
    assert dist.compute_logp(dist_params, sample).shape == (5, 1)
    assert dist.compute_entropy(dist_params).shape == (5, 1)
    assert torch.allclose(dist.compute_kl_div(dist_params, dist_params), torch.zeros(5, 1), atol=1e-6)


def test_one_hot_categorical_dist_keeps_distribution_math_in_float32():
    dist = cusrl.OneHotCategoricalDist(input_dim=4, output_dim=3).half()
    latent = torch.randn(5, 4, dtype=torch.float16)

    dist_params = dist(latent)
    sample, logp = dist.sample_from_dist(dist_params)
    half_dist_params = {"logits": dist_params["logits"].half()}

    assert dist_params["logits"].dtype == torch.float32
    assert sample.dtype == torch.float32
    assert logp.dtype == torch.float32
    assert dist.compute_logp(half_dist_params, sample.half()).dtype == torch.float32
    assert dist.compute_entropy(half_dist_params).dtype == torch.float32
    assert dist.compute_kl_div(half_dist_params, half_dist_params).dtype == torch.float32


def test_normal_distributions_reject_non_positive_init_std():
    with pytest.raises(ValueError, match="'init_std' must be positive"):
        cusrl.NormalDist(input_dim=4, output_dim=2, init_std=0.0)
    with pytest.raises(ValueError, match="'init_std' must be positive"):
        cusrl.AdaptiveNormalDist(input_dim=4, output_dim=2, init_std=-1.0)


def test_one_hot_categorical_dist_disables_autocast_for_logits_head():
    torch.manual_seed(0)
    dist = cusrl.OneHotCategoricalDist(input_dim=4, output_dim=3)
    latent = torch.randn(5, 4)
    expected_logits = dist.mean_head(latent)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        dist_params = dist(latent)

    assert dist_params["logits"].dtype == torch.float32
    assert torch.equal(dist_params["logits"], expected_logits)
