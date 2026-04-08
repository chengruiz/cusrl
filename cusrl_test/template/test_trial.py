from pathlib import Path

import pytest
import torch

import cusrl


def _write_checkpoint(trial_dir: Path, iteration: int, value: int = 1) -> Path:
    checkpoint_path = trial_dir / "ckpt" / f"ckpt_{iteration}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"value": value}, checkpoint_path)
    return checkpoint_path


@pytest.mark.parametrize(
    ("experiment_name", "expected_environment_name", "expected_algorithm_name"),
    [
        ("MountainCar-v0_ppo", "MountainCar-v0", "ppo"),
        ("MountainCar-v0:ppo", "MountainCar-v0", "ppo"),
    ],
)
def test_trial_parses_new_and_legacy_experiment_names(
    tmp_path,
    experiment_name,
    expected_environment_name,
    expected_algorithm_name,
):
    trial_dir = tmp_path / experiment_name / "trial"
    _write_checkpoint(trial_dir, iteration=0)

    trial = cusrl.Trial(trial_dir, verbose=False)

    assert trial.experiment_name == experiment_name
    assert trial.environment_name == expected_environment_name
    assert trial.algorithm_name == expected_algorithm_name


def test_trial_loads_latest_trial_from_experiment_directory_via_latest_symlink(tmp_path):
    experiment_dir = tmp_path / "MountainCar-v0_ppo"
    latest_trial_dir = experiment_dir / "2026-03-26-12-00-00_run"
    _write_checkpoint(latest_trial_dir, iteration=2, value=2)
    (experiment_dir / "latest").symlink_to(latest_trial_dir.name, target_is_directory=True)

    trial = cusrl.Trial(experiment_dir, verbose=False)

    assert trial.home == latest_trial_dir.absolute()
    assert trial.iteration == 2
    assert trial.all_iterations == [2]


def test_trial_rejects_experiment_directory_without_latest_symlink(tmp_path):
    experiment_dir = tmp_path / "MountainCar-v0_ppo"
    _write_checkpoint(experiment_dir / "2026-03-26-12-00-00_run", iteration=2, value=2)

    with pytest.raises(FileNotFoundError, match="is not a valid trial directory"):
        cusrl.Trial(experiment_dir, verbose=False)


def test_trial_ignores_non_checkpoint_files_in_checkpoint_directory(tmp_path):
    trial_dir = tmp_path / "MountainCar-v0_ppo" / "trial"
    ckpt_dir = trial_dir / "ckpt"
    _write_checkpoint(trial_dir, iteration=1)
    (ckpt_dir / "notes.txt").write_text("not a checkpoint")
    (ckpt_dir / "tmp").mkdir()

    trial = cusrl.Trial(trial_dir, verbose=False)

    assert trial.all_iterations == [1]
    assert trial.iteration == 1


def test_trial_raises_when_checkpoint_directory_has_no_checkpoints(tmp_path):
    trial_dir = tmp_path / "MountainCar-v0_ppo" / "trial"
    ckpt_dir = trial_dir / "ckpt"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "notes.txt").write_text("not a checkpoint")

    with pytest.raises(FileNotFoundError, match="No checkpoint files found"):
        cusrl.Trial(trial_dir, verbose=False)


def test_trial_raises_when_checkpoint_file_is_outside_trial_layout(tmp_path):
    checkpoint_path = tmp_path / "ckpt_1.pt"
    torch.save({"value": 1}, checkpoint_path)

    with pytest.raises(FileNotFoundError, match=r"/ckpt"):
        cusrl.Trial(checkpoint_path, verbose=False)


@pytest.mark.parametrize(
    ("filename", "expected_iteration"),
    [
        ("ckpt_0.pt", 0),
        ("ckpt_12.pt", 12),
        ("ckpt_a.pt", None),
        ("foo_12.pt", None),
        ("ckpt_12.pth", None),
        ("ckpt_12.pt.bak", None),
    ],
)
def test_get_ckpt_iteration_uses_full_filename_regex(filename, expected_iteration):
    assert cusrl.Trial._get_ckpt_iteration(Path(filename)) == expected_iteration


def test_is_checkpoint_file_uses_full_filename_regex(tmp_path):
    valid = tmp_path / "ckpt_12.pt"
    invalid_alpha = tmp_path / "ckpt_a.pt"
    invalid_suffix = tmp_path / "ckpt_12.pth"
    invalid_prefix = tmp_path / "foo_12.pt"

    valid.write_text("ok")
    invalid_alpha.write_text("ok")
    invalid_suffix.write_text("ok")
    invalid_prefix.write_text("ok")

    assert cusrl.Trial._is_checkpoint_file(valid) is True
    assert cusrl.Trial._is_checkpoint_file(invalid_alpha) is False
    assert cusrl.Trial._is_checkpoint_file(invalid_suffix) is False
    assert cusrl.Trial._is_checkpoint_file(invalid_prefix) is False
    assert cusrl.Trial._is_checkpoint_file(tmp_path / "ckpt_13.pt") is False
