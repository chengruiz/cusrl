import pytest

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
