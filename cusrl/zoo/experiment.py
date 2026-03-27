import pickle
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from cusrl.template import Environment, LoggerFactoryLike, Player, Trainer, Trial
from cusrl.template.agent import AgentFactory
from cusrl.template.player import PlayerHook

__all__ = ["ExperimentSpec"]


@dataclass(slots=True)
class ExperimentSpec:
    environment_name: str
    algorithm_name: str
    agent_factory_cls: type[AgentFactory]
    agent_factory_kwargs: dict[str, Any]
    training_env_factory: Callable[..., Environment]
    training_env_args: tuple[Any, ...] = None
    training_env_kwargs: dict[str, Any] = field(default_factory=dict)
    trainer_callbacks: Iterable[Callable[["Trainer"], None]] = ()
    player_class: type | None = None
    playing_env_factory: Callable[..., Environment] = None
    playing_env_args: tuple[Any, ...] = None
    playing_env_kwargs: dict[str, Any] = None
    player_hooks: Iterable[PlayerHook] = ()
    num_iterations: int = 1000
    save_interval: int = 50

    def __post_init__(self):
        disallowed = (":", "_", "/", "\\")
        if any(token in self.environment_name for token in disallowed):
            raise ValueError(f"'environment_name' cannot contain ':', '_', '/', or '\\'; got '{self.environment_name}'")
        if any(token in self.algorithm_name for token in disallowed):
            raise ValueError(f"'algorithm_name' cannot contain ':', '_', '/', or '\\'; got '{self.algorithm_name}'")
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
        return f"{self.environment_name}_{self.algorithm_name}"

    def make_agent_factory(self, override_kwargs: dict[str, Any] | None = None, **kwargs) -> AgentFactory:
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
            print(f"Failed to pickle the experiment spec: {error}")
        trainer = Trainer(
            environment=partial(self.make_training_env, environment_kwargs),
            agent_factory=self.make_agent_factory(agent_factory_kwargs),
            logger_factory=logger_factory,
            num_iterations=self.num_iterations if num_iterations is None else num_iterations,
            init_iteration=init_iteration,
            save_interval=self.save_interval if save_interval is None else save_interval,
            checkpoint_path=checkpoint_path,
            verbose=verbose,
            callbacks=self.trainer_callbacks,
        )
        trainer.dump_object(serialized, "experiment_spec")
        return trainer

    def make_player(
        self,
        environment_kwargs: dict[str, Any] | None = None,
        agent_factory_kwargs: dict[str, Any] | None = None,
        checkpoint_path: str | Trial | None = None,
        num_steps: int | None = None,
        num_episodes: int | None = None,
        timestep: float | None = None,
        deterministic: bool = True,
        verbose: bool = True,
    ):
        player_class = self.player_class or Player
        return player_class(
            environment=partial(self.make_playing_env, environment_kwargs),
            agent=self.make_agent_factory(agent_factory_kwargs),
            checkpoint_path=checkpoint_path,
            num_steps=num_steps,
            num_episodes=num_episodes,
            timestep=timestep,
            deterministic=deterministic,
            verbose=verbose,
            hooks=self.player_hooks,
        )
