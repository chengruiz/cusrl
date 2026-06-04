import types

from cusrl.logger.swanlab_logger import Swanlab
from cusrl.logger.wandb_logger import Wandb


def test_wandb_logger_uses_optional_provider_when_available(tmp_path, monkeypatch):
    calls = {}

    fake_wandb = types.SimpleNamespace(
        init=lambda **kwargs: calls.setdefault("init", kwargs) or object(),
        log=lambda data, step: calls.setdefault("log", (data, step)),
    )
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)

    logger = Wandb(tmp_path, name="run", add_datetime_prefix=False, mode="disabled")
    logger.log({"reward": 1.0}, iteration=5)

    assert calls["init"]["name"] == "run"
    assert calls["init"]["mode"] == "disabled"
    assert calls["log"] == ({"reward": 1.0}, 5)


def test_swanlab_logger_uses_optional_provider_when_available(tmp_path, monkeypatch):
    calls = {}

    fake_swanlab = types.SimpleNamespace(
        init=lambda **kwargs: calls.setdefault("init", kwargs) or object(),
        log=lambda data, step: calls.setdefault("log", (data, step)),
    )
    monkeypatch.setitem(__import__("sys").modules, "swanlab", fake_swanlab)

    logger = Swanlab(tmp_path, name="run", add_datetime_prefix=False, mode="disabled")
    logger.log({"reward": 2.0}, iteration=6)

    assert calls["init"]["experiment_name"] == "run"
    assert calls["init"]["mode"] == "disabled"
    assert calls["log"] == ({"reward": 2.0}, 6)
