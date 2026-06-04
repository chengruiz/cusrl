import importlib

import pytest

from cusrl.zoo.experiment import ExperimentSpec

registry_module = importlib.import_module("cusrl.zoo.registry")


@pytest.fixture
def isolated_registry(monkeypatch):
    monkeypatch.setattr(registry_module, "registry", {})
    monkeypatch.setattr(registry_module, "experiment_modules", [])
    return registry_module.registry


def test_register_experiment_supports_multiple_environment_names(isolated_registry):
    registry_module.register_experiment(
        environment_name=["cartpole", "pendulum"],
        algorithm_name="ppo",
        agent_meta_factory=lambda: object(),
        training_env_factory=lambda name: object(),
    )

    assert sorted(isolated_registry) == ["cartpole_ppo", "pendulum_ppo"]

    with pytest.raises(ValueError, match="already registered"):
        registry_module.register_experiment(
            environment_name="cartpole",
            algorithm_name="ppo",
            agent_meta_factory=lambda: object(),
            training_env_factory=lambda name: object(),
        )


def test_get_experiment_reports_available_names(isolated_registry):
    registry_module.register_experiment(
        environment_name="cartpole",
        algorithm_name="ppo",
        agent_meta_factory=lambda: object(),
        training_env_factory=lambda name: object(),
    )

    assert registry_module.get_experiment("cartpole", "ppo").experiment_name == "cartpole_ppo"

    with pytest.raises(ValueError, match="Available experiments:\\n  - cartpole_ppo"):
        registry_module.get_experiment("missing", "ppo")


def test_add_and_load_experiment_modules_imports_registered_modules(tmp_path, monkeypatch, isolated_registry):
    module_path = tmp_path / "registered_module.py"
    module_path.write_text(
        "from cusrl.zoo import register_experiment\n"
        "register_experiment('dummy', 'ppo', lambda: object(), lambda name: object())\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    registry_module.add_experiment_modules("registered_module")
    registry_module.load_experiment_modules()

    assert sorted(isolated_registry) == ["dummy_ppo"]
    assert registry_module.experiment_modules == []


def test_experiment_spec_builds_training_playing_and_benchmarking_factories():
    agent_factory = object()

    def make_agent(**kwargs):
        assert kwargs == {"width": 32}
        return agent_factory

    def make_config(environment_name, *, variant):
        return {"environment_name": environment_name, "variant": variant}

    def make_environment(environment_name, config=None, **kwargs):
        return environment_name, config, kwargs

    spec = ExperimentSpec(
        environment_name="cartpole",
        algorithm_name="ppo",
        agent_meta_factory=make_agent,
        agent_meta_factory_kwargs={"width": 32},
        training_env_factory=make_environment,
        training_env_config_factory=make_config,
        training_env_config_factory_kwargs={"variant": "train"},
        training_env_factory_kwargs={"num_envs": 4},
        num_iterations=7,
        checkpoint_interval=3,
    )

    training_factory = spec.to_training_factory()
    playing_factory = spec.to_playing_factory()
    benchmarking_factory = spec.to_benchmarking_factory()

    assert training_factory.experiment_name == "cartpole_ppo"
    assert training_factory.agent_factory is agent_factory
    assert training_factory.num_iterations == 7
    assert training_factory.checkpoint_interval == 3
    assert training_factory.make_environment() == (
        "cartpole",
        {"environment_name": "cartpole", "variant": "train"},
        {"num_envs": 4},
    )
    assert playing_factory.make_environment() == training_factory.make_environment()
    assert benchmarking_factory.make_environment() == training_factory.make_environment()


@pytest.mark.parametrize("environment_name, algorithm_name", [("bad_env", "ppo"), ("cartpole", "bad/ppo")])
def test_experiment_spec_rejects_names_that_break_experiment_name_parsing(environment_name, algorithm_name):
    with pytest.raises(ValueError, match="cannot contain"):
        ExperimentSpec(
            environment_name=environment_name,
            algorithm_name=algorithm_name,
            agent_meta_factory=lambda: object(),
            training_env_factory=lambda name: object(),
        )
