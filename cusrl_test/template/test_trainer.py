import torch

from cusrl.template.trainer import EnvironmentStats


def test_environment_stats_tracks_rewards_on_configured_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    stats = EnvironmentStats(num_envs=2, reward_dim=1, buffer_size=2, device=device)
    reward = torch.tensor([[1.0], [3.0]], device=device)

    stats.track_step(reward)

    assert stats.device == device
    assert stats.reward.device == device
    assert stats.episode_rew.device == device
    assert torch.equal(stats.reward.cpu(), torch.tensor([2.0]))


def test_environment_stats_state_dict_preserves_stats_device_and_clones_tensors():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    stats = EnvironmentStats(num_envs=2, reward_dim=1, buffer_size=2, device=device)
    stats.track_step(torch.tensor([[1.0], [3.0]], device=device))

    state_dict = stats.state_dict()

    assert state_dict["episode_rew"].device == device
    assert state_dict["episode_len"].device == device
    assert state_dict["rew_buffer"].device == device
    assert state_dict["len_buffer"].device == device
    assert state_dict["episode_rew"] is not stats.episode_rew
    assert state_dict["episode_len"] is not stats.episode_len
