import torch

import subprocess

from cusrl.template.trainer import EnvironmentStats, save_version_info


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


def test_environment_stats_track_episode_truncates_to_buffer_size():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    stats = EnvironmentStats(num_envs=5, reward_dim=1, buffer_size=2, device=device)
    stats.track_step(torch.tensor([[10.0], [20.0], [30.0], [40.0], [50.0]], device=device))

    stats.track_episode(slice(None))

    assert stats.num_episodes == 5
    assert torch.equal(stats.rew_buffer.cpu(), torch.tensor([[50.0], [40.0]]))
    assert stats.mean_episode_reward == 45.0
    assert stats.mean_episode_length == 1.0
    assert torch.all(stats.episode_rew == 0)
    assert torch.all(stats.episode_len == 0)


def test_save_version_info_handles_non_git_and_empty_git_repo(tmp_path):
    non_git_dir = tmp_path / "non_git"
    non_git_dir.mkdir()
    out_non_git = tmp_path / "out_non_git"
    save_version_info(out_non_git, workspace=non_git_dir)
    assert not out_non_git.exists()

    empty_repo = tmp_path / "empty_repo"
    empty_repo.mkdir()
    subprocess.run(["git", "init"], cwd=empty_repo, check=True, capture_output=True)
    out_empty = tmp_path / "out_empty"
    save_version_info(out_empty, workspace=empty_repo)

    assert (out_empty / "workspace.txt").read_text() == str(empty_repo.resolve())
    assert (out_empty / "git_log.txt").read_text() == ""
    assert (out_empty / "git_diff.patch").read_text() == ""
    assert (out_empty / "version.txt").read_text() == "unknown"
    assert (out_empty / "git_status.txt").exists()
