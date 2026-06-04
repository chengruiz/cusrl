import pytest
import torch

from cusrl.utils.metrics import Metrics


def test_metrics_records_weighted_means_and_prefixes_summary_keys():
    metrics = Metrics()

    metrics.record({"loss": torch.tensor([1.0, 3.0])}, accuracy=0.5, ignored=None)
    metrics.record(loss=torch.tensor([5.0, 7.0, 9.0]))

    assert len(metrics) == 2
    assert metrics["loss"].count == 5
    assert metrics.summary("train") == pytest.approx({
        "train/loss": 5.0,
        "train/accuracy": 0.5,
    })


def test_metrics_ignores_empty_values_and_can_clear():
    metrics = Metrics()

    metrics.record(empty=torch.tensor([]), missing=None)
    assert len(metrics) == 0

    metrics.record(value=[1.0, 2.0])
    metrics.clear()
    assert list(metrics.items()) == []


def test_metrics_reports_conversion_errors_with_metric_name():
    metrics = Metrics()

    with pytest.raises(ValueError, match="bad_metric"):
        metrics.record(bad_metric=object())
