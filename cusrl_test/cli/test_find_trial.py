from pathlib import Path

import pytest

from cusrl.cli import find_trial


def _make_trial(experiment_dir: Path, trial_name: str):
    ckpt_dir = experiment_dir / trial_name / "ckpt"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "ckpt_0.pt").write_text("checkpoint")


def test_name_filter_matches_exact_run_name_after_timestamp(tmp_path, capsys):
    experiment_dir = tmp_path / "Env_Alg"
    _make_trial(experiment_dir, "2026-03-26-12-00-00_run")
    _make_trial(experiment_dir, "2026-03-26-13-00-00_run-extra")
    _make_trial(experiment_dir, "2026-03-26-14-00-00_other-run")

    find_trial.main([
        "--environment",
        "Env",
        "--algorithm",
        "Alg",
        "--log-dir",
        str(tmp_path),
        "--name",
        "run",
        "--basename",
        "--list",
    ])

    assert capsys.readouterr().out.splitlines() == ["2026-03-26-12-00-00_run"]


def test_name_filter_matches_exact_directory_name_without_timestamp(tmp_path, capsys):
    experiment_dir = tmp_path / "Env_Alg"
    _make_trial(experiment_dir, "run")
    _make_trial(experiment_dir, "run-extra")

    find_trial.main([
        "--environment",
        "Env",
        "--algorithm",
        "Alg",
        "--log-dir",
        str(tmp_path),
        "--name",
        "run",
        "--basename",
        "--list",
    ])

    assert capsys.readouterr().out.splitlines() == ["run"]


def test_name_filter_error_describes_exact_name(tmp_path):
    experiment_dir = tmp_path / "Env_Alg"
    _make_trial(experiment_dir, "2026-03-26-12-00-00_run-extra")

    with pytest.raises(FileNotFoundError, match="with name 'run'"):
        find_trial.main([
            "--environment",
            "Env",
            "--algorithm",
            "Alg",
            "--log-dir",
            str(tmp_path),
            "--name",
            "run",
        ])
