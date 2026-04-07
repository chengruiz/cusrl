from datetime import datetime

import pytest
import torch

import cusrl


class CaptureLogger(cusrl.Logger):
    def __init__(self, *args, **kwargs):
        self.calls = []
        super().__init__(*args, **kwargs)

    def _log_impl(self, data: dict[str, float], iteration: int):
        self.calls.append((iteration, data))


def test_logger_interval_batches_by_call_count(tmp_path):
    logger = CaptureLogger(tmp_path, name="run", add_datetime_prefix=False, interval=3)

    for iteration in range(6):
        logger.log({"value": float(iteration)}, iteration)

    assert logger.calls == [
        (2, {"value": 1.0}),
        (5, {"value": 4.0}),
    ]


def test_logger_interval_batches_by_call_count_after_resume(tmp_path):
    logger = CaptureLogger(tmp_path, name="run", add_datetime_prefix=False, interval=3)

    for iteration in range(5, 11):
        logger.log({"value": float(iteration)}, iteration)

    assert logger.calls == [
        (7, {"value": 6.0}),
        (10, {"value": 9.0}),
    ]


@pytest.mark.parametrize("logger_type", [None, "none"])
def test_make_logger_factory_preserves_add_datetime_prefix(tmp_path, logger_type):
    logger_factory = cusrl.make_logger_factory(
        logger_type=logger_type,
        log_dir=tmp_path,
        name="run",
        add_datetime_prefix=False,
    )

    assert logger_factory is not None
    logger = logger_factory()

    assert logger.log_dir == tmp_path / "run"
    assert (tmp_path / "latest").is_symlink()
    assert (tmp_path / "latest").resolve() == logger.log_dir


def test_logger_name_none_writes_directly_under_log_dir(tmp_path):
    log_dir = tmp_path / "trial"
    logger = cusrl.Logger(log_dir, name=None)

    logger.save_checkpoint({"value": 1}, 0)
    trial = cusrl.Trial(logger.log_dir, verbose=False)

    assert logger.log_dir == log_dir
    assert not (tmp_path / "latest").exists()
    assert trial.home == logger.log_dir
    assert trial.all_iterations == [0]


def test_logger_uses_single_underscore_for_datetime_prefix(tmp_path, monkeypatch):
    class FixedDatetime(datetime):
        @classmethod
        def now(cls):
            return cls(2026, 3, 26, 12, 34, 56)

    monkeypatch.setattr("cusrl.template.logger.datetime", FixedDatetime)

    logger = cusrl.Logger(tmp_path, name="run")

    assert logger.name == "2026-03-26-12-34-56_run"
    assert logger.log_dir == tmp_path / "2026-03-26-12-34-56_run"


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
    ckpt_dir = trial_dir / "ckpt"
    ckpt_dir.mkdir(parents=True)
    torch.save({"value": 1}, ckpt_dir / "ckpt_0.pt")

    trial = cusrl.Trial(trial_dir, verbose=False)

    assert trial.experiment_name == experiment_name
    assert trial.environment_name == expected_environment_name
    assert trial.algorithm_name == expected_algorithm_name


def test_experiment_spec_uses_single_underscore_name_separator():
    spec = cusrl.zoo.ExperimentSpec(
        environment_name="MountainCar-v0",
        algorithm_name="ppo",
        agent_meta_factory=cusrl.preset.PpoAgentFactory,
        training_env_factory=lambda *args, **kwargs: None,
    )

    assert spec.experiment_name == "MountainCar-v0_ppo"
