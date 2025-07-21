import pickle
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from cusrl.template import Agent, Environment, LoggerFactoryLike, Player, Trainer, Trial

__all__ = ["ExperimentSpec"]


@dataclass(slots=True)
class ExperimentSpec:
    environment_name: str
    algorithm_name: str
    agent_factory_cls: type[Agent.Factory]
    agent_factory_kwargs: dict[str, Any]
    training_env_factory: Callable[..., Environment]
    training_env_args: tuple[Any, ...] = None
    training_env_kwargs: dict[str, Any] = field(default_factory=dict)
    playing_env_factory: Callable[..., Environment] = None
    playing_env_args: tuple[Any, ...] = None
    playing_env_kwargs: dict[str, Any] = None
    num_iterations: int = 1000
    save_interval: int = 50
    callbacks: Iterable[Callable[["Trainer"], None]] = ()

    def __post_init__(self):
        if ":" in self.environment_name or "/" in self.environment_name or "\\" in self.algorithm_name:
            raise ValueError(f"environment_name '{self.environment_name}' cannot contain ':', '/', or '\\'.")
        if ":" in self.algorithm_name or "/" in self.algorithm_name or "\\" in self.algorithm_name:
            raise ValueError(f"algorithm_name '{self.algorithm_name}' cannot contain ':', '/', or '\\'.")
        if self.training_env_args is None:
            self.training_env_args = (self.environment_name,)
        if self.playing_env_factory is None:
            self.playing_env_factory = self.training_env_factory
        if self.playing_env_args is None:
            self.playing_env_args = self.training_env_args
        if self.playing_env_kwargs is None:
            self.playing_env_kwargs = self.training_env_kwargs.copy()

    @property
    def name(self) -> str:
        return f"{self.environment_name}:{self.algorithm_name}"

    def make_agent_factory(self, override_kwargs: dict[str, Any] | None = None, **kwargs) -> Agent.Factory:
        agent_factory_kwargs = self.agent_factory_kwargs | (override_kwargs or {}) | kwargs
        return self.agent_factory_cls(**agent_factory_kwargs)

    def make_training_env(self, override_kwargs: dict[str, Any] | None = None, **kwargs) -> Environment:
        training_env_kwargs = self.training_env_kwargs | (override_kwargs or {}) | kwargs
        return self.training_env_factory(*self.training_env_args, **training_env_kwargs)

    def make_playing_env(self, override_kwargs: dict[str, Any] | None = None, **kwargs) -> Environment:
        playing_env_kwargs = self.playing_env_kwargs | (override_kwargs or {}) | kwargs
        return self.playing_env_factory(*self.playing_env_args, **playing_env_kwargs)

    def make_trainer(
        self,
        environment_kwargs: dict[str, Any] | None = None,
        agent_factory_kwargs: dict[str, Any] | None = None,
        logger_factory: LoggerFactoryLike | None = None,
        num_iterations: int | None = None,
        init_iteration: int | None = None,
        save_interval: int | None = None,
        checkpoint_path: str | None = None,
        verbose: bool = True,
    ) -> Trainer:
        try:
            serialized = pickle.dumps(self)
        except Exception as error:
            serialized = None
            print(f"Failed to pickle experiment spec due to: {error}")
        trainer = Trainer(
            environment=partial(self.make_training_env, environment_kwargs),
            agent_factory=self.make_agent_factory(agent_factory_kwargs),
            logger_factory=logger_factory,
            num_iterations=num_iterations or self.num_iterations,
            init_iteration=init_iteration,
            save_interval=save_interval or self.save_interval,
            checkpoint_path=checkpoint_path,
            verbose=verbose,
            callbacks=self.callbacks,
        )
        trainer.dump_object(serialized, "experiment_spec")
        return trainer

    def make_player(
        self,
        environment_kwargs: dict[str, Any] | None = None,
        agent_factory_kwargs: dict[str, Any] | None = None,
        checkpoint_path: str | Trial | None = None,
        num_steps: int | None = None,
        timestep: float | None = None,
        deterministic: bool = True,
        verbose: bool = True,
        hooks: Iterable[Player.Hook] = (),
    ):
        return Player(
            environment=partial(self.make_playing_env, environment_kwargs),
            agent=self.make_agent_factory(agent_factory_kwargs),
            checkpoint_path=checkpoint_path,
            num_steps=num_steps,
            timestep=timestep,
            deterministic=deterministic,
            verbose=verbose,
            hooks=hooks,
        )
