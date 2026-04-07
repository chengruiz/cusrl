import torch

import cusrl
from cusrl.nn.module.distribution import StddevVector


def test_stddev_vector_fill_and_clamp_apply_bijector_bounds():
    std = StddevVector(output_dim=3, bijector="exp")
    latent = torch.zeros(2, 5)

    assert torch.allclose(std(latent), torch.ones(2, 3))

    std.fill_(0.5)
    assert torch.allclose(std(latent), torch.full((2, 3), 0.5))

    std.clamp_(lb=0.6, ub=0.7, indices=slice(0, 2))
    output = std(latent)
    assert torch.allclose(output[:, :2], torch.full((2, 2), 0.6))
    assert torch.allclose(output[:, 2], torch.full((2,), 0.5))


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

    dist.set_std(0.75)
    dist.clamp_std(lb=0.8, indices=slice(0, 1))
    clamped_std = dist(latent)["std"]
    assert torch.allclose(clamped_std[:, 0], torch.full((3,), 0.8))
    assert torch.allclose(clamped_std[:, 1], torch.full((3,), 0.75))


def test_adaptive_normal_dist_backward_flag_detaches_std_branch_from_latent():
    torch.manual_seed(0)
    dist = cusrl.AdaptiveNormalDist(input_dim=3, output_dim=2, bijector=None, backward=False)
    latent = torch.randn(4, 3, requires_grad=True)

    expected_grad = torch.autograd.grad(dist.mean_head(latent).sum(), latent, retain_graph=True)[0]
    latent.grad = None

    dist_params = dist(latent)
    (dist_params["mean"].sum() + dist_params["std"].sum()).backward()

    assert torch.allclose(latent.grad, expected_grad)


def test_one_hot_categorical_dist_returns_one_hot_actions_and_statistics():
    dist = cusrl.OneHotCategoricalDist(input_dim=4, output_dim=3)
    dist.mean_head.weight.data.zero_()
    dist.mean_head.bias.data.copy_(torch.tensor([1.0, 3.0, 2.0]))
    latent = torch.zeros(5, 4)

    dist_params = dist(latent)
    mode = dist.determine(latent)
    sample, logp = dist.sample_from_dist(dist_params)

    assert dist_params["logit"].shape == (5, 3)
    assert torch.equal(mode, torch.tensor([[0, 1, 0]]).repeat(5, 1))
    assert sample.shape == (5, 3)
    assert torch.equal(sample.sum(dim=-1), torch.ones(5))
    assert logp.shape == (5, 1)
    assert dist.compute_logp(dist_params, sample).shape == (5, 1)
    assert dist.compute_entropy(dist_params).shape == (5, 1)
    assert torch.allclose(dist.compute_kl_div(dist_params, dist_params), torch.zeros(5, 1), atol=1e-6)
