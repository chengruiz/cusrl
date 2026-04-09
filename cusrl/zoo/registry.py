import importlib
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from cusrl.template import AgentFactory, Environment, Player, PlayerHook, TrainerHook
from cusrl.zoo.experiment import ExperimentSpec

__all__ = [
    "ExperimentSpec",
    "add_experiment_modules",
    "get_experiment",
    "load_experiment_modules",
    "register_experiment",
    "registry",
]


registry: dict[str, ExperimentSpec] = {}
experiment_modules = [
    "cusrl.zoo.gym",
    "cusrl.zoo.isaaclab",
    "cusrl.zoo.robot_lab",
    "cusrl.zoo.mjlab",
]


def register_experiment(
    environment_name: str | Sequence[str],
    algorithm_name: str,
    agent_meta_factory: Callable[..., AgentFactory],
    training_env_factory: Callable[..., Environment],
    agent_meta_factory_kwargs: dict[str, Any] | None = None,
    training_env_config_factory: Callable[..., Any] | None = None,
    training_env_config_factory_kwargs: dict[str, Any] | None = None,
    training_env_factory_kwargs: dict[str, Any] | None = None,
    trainer_hooks: Iterable[TrainerHook] = (),
    player_factory: Callable[..., Player] = Player,
    playing_env_factory: Callable[..., Environment] | None = None,
    playing_env_config_factory: Callable[..., Any] | None = None,
    playing_env_config_factory_kwargs: dict[str, Any] | None = None,
    playing_env_factory_kwargs: dict[str, Any] | None = None,
    player_hooks: Iterable[PlayerHook] = (),
    benchmarker_factory: Callable[..., Player] = Player,
    benchmarking_env_factory: Callable[..., Environment] | None = None,
    benchmarking_env_config_factory: Callable[..., Any] | None = None,
    benchmarking_env_config_factory_kwargs: dict[str, Any] | None = None,
    benchmarking_env_factory_kwargs: dict[str, Any] | None = None,
    benchmarking_hooks: Iterable[PlayerHook] = (),
    num_iterations: int = 1000,
    checkpoint_interval: int = 50,
):
    if isinstance(environment_name, str):
        environment_name = [environment_name]
    for env_name in environment_name:
        spec = ExperimentSpec(
            environment_name=env_name,
            algorithm_name=algorithm_name,
            agent_meta_factory=agent_meta_factory,
            agent_meta_factory_kwargs=agent_meta_factory_kwargs or {},
            training_env_factory=training_env_factory,
            training_env_config_factory=training_env_config_factory,
            training_env_config_factory_kwargs=training_env_config_factory_kwargs or {},
            training_env_factory_kwargs=training_env_factory_kwargs or {},
            trainer_hooks=tuple(trainer_hooks),
            player_factory=player_factory,
            playing_env_factory=playing_env_factory,
            playing_env_config_factory=playing_env_config_factory,
            playing_env_config_factory_kwargs=playing_env_config_factory_kwargs,
            playing_env_factory_kwargs=playing_env_factory_kwargs,
            player_hooks=tuple(player_hooks),
            benchmarker_factory=benchmarker_factory,
            benchmarking_env_factory=benchmarking_env_factory,
            benchmarking_env_config_factory=benchmarking_env_config_factory,
            benchmarking_env_config_factory_kwargs=benchmarking_env_config_factory_kwargs,
            benchmarking_env_factory_kwargs=benchmarking_env_factory_kwargs,
            benchmarker_hooks=tuple(benchmarking_hooks),
            num_iterations=num_iterations,
            checkpoint_interval=checkpoint_interval,
        )
        if spec.experiment_name in registry:
            raise ValueError(f"Experiment '{spec.experiment_name}' is already registered")
        registry[spec.experiment_name] = spec


def add_experiment_modules(*lib: str):
    experiment_modules.extend(lib)


def load_experiment_modules():
    """Load all registered experiment modules."""
    for module in experiment_modules:
        try:
            importlib.import_module(module)
        except ImportError as error:
            raise ImportError(f"Failed to import experiment module '{module}'") from error
    experiment_modules.clear()


def get_experiment(environment_name: str, algorithm_name: str) -> ExperimentSpec:
    load_experiment_modules()

    key = f"{environment_name}_{algorithm_name}"
    if key not in registry:
        all_experiments = "".join([f"\n  - {experiment_name}" for experiment_name in sorted(registry.keys())])
        raise ValueError(f"Experiment '{key}' is not registered. Available experiments:{all_experiments}")
    return registry[key]
