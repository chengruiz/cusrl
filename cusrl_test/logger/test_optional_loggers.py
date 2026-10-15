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


def test_optional_loggers_use_resolved_name_and_avoid_empty_string(tmp_path, monkeypatch):
    wandb_calls = {}
    swanlab_calls = {}

    monkeypatch.setitem(
        __import__("sys").modules,
        "wandb",
        types.SimpleNamespace(init=lambda **kwargs: wandb_calls.setdefault("init", kwargs) or object()),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "swanlab",
        types.SimpleNamespace(init=lambda **kwargs: swanlab_calls.setdefault("init", kwargs) or object()),
    )

    wandb_logger = Wandb(tmp_path / "wandb_prefixed", name="")
    swanlab_logger = Swanlab(tmp_path / "swanlab_prefixed", name="")
    assert wandb_calls["init"]["name"] == wandb_logger.name
    assert swanlab_calls["init"]["experiment_name"] == swanlab_logger.name

    wandb_calls.clear()
    swanlab_calls.clear()
    Wandb(tmp_path / "wandb_empty", name="", add_datetime_prefix=False)
    Swanlab(tmp_path / "swanlab_empty", name="", add_datetime_prefix=False)
    assert wandb_calls["init"]["name"] is None
    assert swanlab_calls["init"]["experiment_name"] is None
