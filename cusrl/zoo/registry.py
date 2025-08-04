import importlib
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from cusrl.template import Agent, Environment, Player, Trainer
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
]


def register_experiment(
    environment_name: str | Sequence[str],
    algorithm_name: str,
    agent_factory_cls: type[Agent.Factory],
    agent_factory_kwargs: dict[str, Any],
    training_env_factory: Callable[..., Environment],
    training_env_args: tuple[Any, ...] | None = None,
    training_env_kwargs: dict[str, Any] | None = None,
    trainer_callbacks: Iterable[Callable[["Trainer"], None]] = (),
    playing_env_factory: Callable[..., Environment] | None = None,
    playing_env_args: tuple[Any, ...] | None = None,
    playing_env_kwargs: dict[str, Any] | None = None,
    player_hooks: Iterable[Player.Hook] = (),
    num_iterations: int = 1000,
    save_interval: int = 50,
):
    if isinstance(environment_name, str):
        environment_name = [environment_name]
    for env_name in environment_name:
        spec = ExperimentSpec(
            environment_name=env_name,
            algorithm_name=algorithm_name,
            agent_factory_cls=agent_factory_cls,
            agent_factory_kwargs=agent_factory_kwargs,
            training_env_factory=training_env_factory,
            training_env_args=training_env_args,
            training_env_kwargs=training_env_kwargs or {},
            trainer_callbacks=trainer_callbacks,
            playing_env_factory=playing_env_factory,
            playing_env_args=playing_env_args,
            playing_env_kwargs=playing_env_kwargs,
            player_hooks=player_hooks,
            num_iterations=num_iterations,
            save_interval=save_interval,
        )
        if spec.name in registry:
            raise ValueError(f"Experiment '{spec.name}' is already registered.")
        registry[spec.name] = spec


def add_experiment_modules(*lib: str):
    experiment_modules.extend(lib)


def load_experiment_modules():
    """Load all registered experiment modules."""
    for module in experiment_modules:
        try:
            importlib.import_module(module)
        except ImportError as error:
            raise ImportError(f"Failed to import experiment module '{module}'.") from error
    experiment_modules.clear()


def get_experiment(environment_name: str, algorithm_name: str) -> ExperimentSpec:
    load_experiment_modules()

    key = f"{environment_name}:{algorithm_name}"
    if key not in registry:
        all_experiments = "".join([f"\n  - {experiment_name}" for experiment_name in sorted(registry.keys())])
        raise ValueError(f"Experiment '{key}' is not registered. Available experiments are: {all_experiments}.")
    return registry[key]
