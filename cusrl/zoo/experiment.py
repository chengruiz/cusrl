import shlex
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any

import tyro

import cusrl
from cusrl.template import AgentFactory, Environment, Player, PlayerHook, Trainer
from cusrl.utils.tyro_utils import JsonDict

__all__ = ["ExperimentSpec"]


def _make_environment_config(
    factory: Callable[..., Any] | None,
    environment_name: str,
    kwargs: dict[str, Any] | None,
) -> Any:
    if factory is None:
        return None
    return factory(environment_name, **(kwargs or {}))


@dataclass(kw_only=True)
class AgentFactorySpec:
    agent_factory: Annotated[AgentFactory, tyro.conf.arg(name="agent")]
    """Agent factory specification"""


@dataclass(kw_only=True)
class EnvironmentFactorySpec:
    environment_name: Annotated[str, tyro.conf.Suppress]
    environment_factory: Annotated[Callable[..., Environment], tyro.conf.Suppress]
    environment_config: Annotated[Any, tyro.conf.arg(name="env")] = None
    """Environment configuration"""
    environment_args: Annotated[str | None, tyro.conf.arg(name="env_args")] = None
    """Command-line arguments for the environment, as a single string"""
    environment_kwargs: Annotated[JsonDict, tyro.conf.arg(name="env_kwargs")] = field(default_factory=dict)
    """Keyword arguments to pass to the environment factory"""

    def make_environment(self) -> Environment:
        args = [self.environment_name]
        if self.environment_config is not None:
            args.append(self.environment_config)
        kwargs = self.environment_kwargs or {}
        if self.environment_args is not None:
            kwargs["argv"] = shlex.split(self.environment_args)
        return self.environment_factory(*args, **kwargs)


@dataclass(kw_only=True)
class TrainingExperimentFactory(AgentFactorySpec, EnvironmentFactorySpec):
    experiment_name: Annotated[str, tyro.conf.Suppress]
    name: str = ""
    logger: str = "tensorboard"
    log_dir: str = "logs"
    log_interval: int = 1
    num_iterations: int = 1000
    init_iteration: int | None = None
    checkpoint_interval: int = 50
    checkpoint_path: str | None = None
    callbacks: Sequence[Callable[["Trainer"], None]] = ()

    def __call__(self) -> Trainer:
        trainer = Trainer(
            environment=self.make_environment(),
            agent_factory=self.agent_factory,
            logger_factory=cusrl.make_logger_factory(
                self.logger,
                log_dir=f"{self.log_dir}/{self.experiment_name}",
                name=self.name,
                interval=self.log_interval,
            ),
            num_iterations=self.num_iterations,
            init_iteration=self.init_iteration,
            checkpoint_interval=self.checkpoint_interval,
            checkpoint_path=self.checkpoint_path,
            callbacks=self.callbacks,
        )
        return trainer


@dataclass(kw_only=True)
class PlayingExperimentFactory(AgentFactorySpec, EnvironmentFactorySpec):
    player_factory: Annotated[type[Player], tyro.conf.Suppress] = Player
    num_steps: int | None = None
    num_episodes: int | None = None
    timestep: float | None = None
    deterministic: bool = True
    verbose: bool = True
    hooks: Sequence[PlayerHook] = ()

    def __call__(self, checkpoint_path: str | None) -> Player:
        return self.player_factory(
            environment=self.make_environment(),
            agent=self.agent_factory,
            checkpoint_path=checkpoint_path,
            num_steps=self.num_steps,
            num_episodes=self.num_episodes,
            timestep=self.timestep,
            deterministic=self.deterministic,
            verbose=self.verbose,
            hooks=self.hooks,
        )


@dataclass(kw_only=True)
class BenchmarkingExperimentFactory(AgentFactorySpec, EnvironmentFactorySpec):
    benchmarker_factory: Annotated[Callable[..., Player], tyro.conf.Suppress] = Player
    num_steps: int | None = None
    num_episodes: int | None = None
    deterministic: bool = True
    verbose: bool = True
    hooks: Sequence[PlayerHook] = ()

    def __call__(self, checkpoint_path: str | None) -> Player:
        return self.benchmarker_factory(
            environment=self.make_environment(),
            agent=self.agent_factory,
            checkpoint_path=checkpoint_path,
            num_steps=self.num_steps,
            num_episodes=self.num_episodes,
            timestep=0.0,
            deterministic=self.deterministic,
            verbose=self.verbose,
            hooks=self.hooks,
        )


@dataclass(kw_only=True)
class ExperimentSpec:
    environment_name: str
    algorithm_name: str
    agent_meta_factory: Callable[..., AgentFactory]
    agent_meta_factory_kwargs: dict[str, Any] = field(default_factory=dict)

    training_env_factory: Callable[..., Environment]
    training_env_config_factory: Callable[..., Any] | None = None
    training_env_config_factory_kwargs: dict[str, Any] = field(default_factory=dict)
    training_env_factory_kwargs: dict[str, Any] = field(default_factory=dict)
    trainer_callbacks: Sequence[Callable[["Trainer"], None]] = ()
    num_iterations: int = 1000
    checkpoint_interval: int = 50

    player_factory: Callable[..., Player] = Player
    playing_env_factory: Callable[..., Environment] | None = None
    playing_env_config_factory: Callable[..., Any] | None = None
    playing_env_config_factory_kwargs: dict[str, Any] | None = None
    playing_env_factory_kwargs: dict[str, Any] | None = None
    player_hooks: Sequence[PlayerHook] = ()

    benchmarker_factory: Callable[..., Player] = Player
    benchmarking_env_factory: Callable[..., Environment] | None = None
    benchmarking_env_config_factory: Callable[..., Any] | None = None
    benchmarking_env_config_factory_kwargs: dict[str, Any] | None = None
    benchmarking_env_factory_kwargs: dict[str, Any] | None = None
    benchmarker_hooks: Sequence[PlayerHook] = ()

    def __post_init__(self):
        disallowed = (":", "_", "/", "\\")
        if any(token in self.environment_name for token in disallowed):
            raise ValueError(f"'environment_name' cannot contain ':', '_', '/', or '\\'; got '{self.environment_name}'")
        if any(token in self.algorithm_name for token in disallowed):
            raise ValueError(f"'algorithm_name' cannot contain ':', '_', '/', or '\\'; got '{self.algorithm_name}'")

    @property
    def experiment_name(self) -> str:
        return f"{self.environment_name}_{self.algorithm_name}"

    def to_training_factory(self) -> TrainingExperimentFactory:
        return TrainingExperimentFactory(
            experiment_name=self.experiment_name,
            agent_factory=self.agent_meta_factory(**self.agent_meta_factory_kwargs),
            environment_factory=self.training_env_factory,
            environment_name=self.environment_name,
            environment_config=_make_environment_config(
                self.training_env_config_factory,
                self.environment_name,
                self.training_env_config_factory_kwargs,
            ),
            environment_kwargs=self.training_env_factory_kwargs,
            callbacks=self.trainer_callbacks,
            num_iterations=self.num_iterations,
            checkpoint_interval=self.checkpoint_interval,
        )

    def to_playing_factory(self) -> PlayingExperimentFactory:
        playing_env_factory = self.playing_env_factory or self.training_env_factory
        playing_env_config_factory = self.playing_env_config_factory
        playing_env_config_factory_kwargs = self.playing_env_config_factory_kwargs
        playing_env_factory_kwargs = self.playing_env_factory_kwargs

        if playing_env_config_factory is None:
            playing_env_config_factory = self.training_env_config_factory
        if playing_env_config_factory_kwargs is None:
            playing_env_config_factory_kwargs = self.training_env_config_factory_kwargs
        if playing_env_factory_kwargs is None:
            playing_env_factory_kwargs = self.training_env_factory_kwargs

        return PlayingExperimentFactory(
            agent_factory=self.agent_meta_factory(**self.agent_meta_factory_kwargs),
            player_factory=self.player_factory,
            environment_factory=playing_env_factory,
            environment_name=self.environment_name,
            environment_config=_make_environment_config(
                playing_env_config_factory,
                self.environment_name,
                playing_env_config_factory_kwargs,
            ),
            environment_kwargs=playing_env_factory_kwargs,
            hooks=self.player_hooks,
        )

    def to_benchmarking_factory(self) -> BenchmarkingExperimentFactory:
        benchmarking_env_factory = self.benchmarking_env_factory or self.training_env_factory
        benchmarking_env_config_factory = self.benchmarking_env_config_factory
        benchmarking_env_config_factory_kwargs = self.benchmarking_env_config_factory_kwargs
        benchmarking_env_factory_kwargs = self.benchmarking_env_factory_kwargs
        if benchmarking_env_config_factory is None:
            benchmarking_env_config_factory = self.training_env_config_factory
        if benchmarking_env_config_factory_kwargs is None:
            benchmarking_env_config_factory_kwargs = self.training_env_config_factory_kwargs
        if benchmarking_env_factory_kwargs is None:
            benchmarking_env_factory_kwargs = self.training_env_factory_kwargs

        return BenchmarkingExperimentFactory(
            agent_factory=self.agent_meta_factory(**self.agent_meta_factory_kwargs),
            benchmarker_factory=self.benchmarker_factory,
            environment_factory=benchmarking_env_factory,
            environment_name=self.environment_name,
            environment_config=_make_environment_config(
                benchmarking_env_config_factory,
                self.environment_name,
                benchmarking_env_config_factory_kwargs,
            ),
            environment_kwargs=benchmarking_env_factory_kwargs,
            hooks=self.benchmarker_hooks,
        )
