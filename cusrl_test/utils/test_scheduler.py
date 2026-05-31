import pytest

from cusrl.utils.scheduler import (
    CosineAnnealingScheduler,
    PiecewiseLinearScheduler,
    StepScheduler,
    TanhScheduler,
)


def test_step_scheduler_uses_transitions():
    scheduler = StepScheduler(1.0, (10, 0.5), (20, 0.0))

    assert scheduler(0) == 1.0
    assert scheduler(10) == 0.5
    assert scheduler(20) == 0.0


def test_piecewise_linear_scheduler_uses_anchors():
    scheduler = PiecewiseLinearScheduler((0, 1.0), (10, 0.0), (20, 0.5))

    assert scheduler(-1) == 1.0
    assert scheduler(5) == pytest.approx(0.5)
    assert scheduler(15) == pytest.approx(0.25)
    assert scheduler(21) == 0.5


def test_piecewise_linear_scheduler_requires_at_least_two_anchors():
    with pytest.raises(ValueError, match="at least two anchors are required"):
        PiecewiseLinearScheduler((0, 1.0))


def test_cosine_annealing_scheduler_clamps_outside_interval():
    scheduler = CosineAnnealingScheduler((10, 1.0), (20, 0.0))

    assert scheduler(0) == 1.0
    assert scheduler(10) == 1.0
    assert scheduler(20) == 0.0
    assert scheduler(30) == 0.0


def test_cosine_annealing_scheduler_interpolates_between_anchors():
    scheduler = CosineAnnealingScheduler(start=(0, 1.0), end=(10, 0.0))

    assert scheduler(5) == pytest.approx(0.5)


def test_cosine_annealing_scheduler_supports_increasing_values():
    scheduler = CosineAnnealingScheduler((0, 0.0), (10, 1.0))

    assert scheduler(5) == pytest.approx(0.5)


def test_cosine_annealing_scheduler_requires_increasing_steps():
    with pytest.raises(ValueError, match="step coordinates must be strictly increasing"):
        CosineAnnealingScheduler((1, 0.0), (1, 1.0))


def test_tanh_scheduler_uses_start_and_end():
    scheduler = TanhScheduler(start=(0, 1.0), end=(10, 0.0), eta=3.0)

    assert scheduler(-1) == 1.0
    assert scheduler(5) == pytest.approx(0.5)
    assert scheduler(11) == 0.0
