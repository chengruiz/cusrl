import signal
from collections import defaultdict
from collections.abc import Iterable

from tqdm import tqdm
from typing_extensions import Self

from cusrl import utils
from cusrl.template.agent import Agent
from cusrl.template.environment import Environment, get_done_indices, update_observation_and_state
from cusrl.template.trainer import EnvironmentStats
from cusrl.template.trial import Trial
from cusrl.utils.typing import Array, Slice

__all__ = ["Player"]


class PlayerHook:
    player: "Player"
    agent: Agent
    environment: Environment

    def init(self, player: "Player"):
        self.player = player
        self.agent = player.agent
        self.environment = player.environment

    def step(self, step: int, transition: dict[str, Array], metrics: dict[str, float]):
        pass

    def reset(self, indices: Slice):
        pass

    def close(self):
        pass


class PlayerHookComposite(PlayerHook, list[PlayerHook]):
    def init(self, player: "Player"):
        for hook in self:
            hook.init(player)

    def step(self, step: int, transition, metrics):
        for hook in self:
            hook.step(step, transition, metrics)

    def reset(self, indices: Slice):
        for hook in self:
            hook.reset(indices)

    def close(self):
        for hook in self:
            hook.close()


class Player:
    """Orchestrates a playing loop between an Agent and an Environment, also
    manages initialization, checkpoint loading, stepping, and hook callbacks.

    Args:
        environment (Environment | Environment.Factory):
            An Environment instance or a factory that produces one.
        agent (Agent | Agent.Factory):
            An Agent instance or a factory that produces one for the given
            environment.
        checkpoint_path (str | Trial | None, optional):
            Path to a saved checkpoint or a :class:`Trial` object. If provided,
            loads the states of the agent and the environment states from the
            checkpoint.
        num_steps (int | None, optional):
            Maximum number of steps to execute. If ``None``, runs indefinitely.
        timestep (float | None, optional):
            Time interval between steps in seconds. Defaults to
            ``environment.spec.timestep`` if not provided.
        deterministic (bool, optional):
            Whether to run the agent in deterministic mode. If ``False``, the
            agent will sample actions stochastically. Defaults to ``True``.
        verbose (bool, optional):
            Whether to enable verbose logging. Defaults to ``True``.
        hooks (Iterable[PlayerHook], optional):
            A sequence of PlayerHook classes or instances to be initialized and
            called at each step and reset event.

    Methods:
        register_hook(hook: PlayerHook) -> Self
            Register and initialize an additional hook to be called during play.
        run_playing_loop() -> dict[str, float]
            Reset the environment, then repeatedly:
            - Query the agent for an action;
            - Step the environment;
            - Update the agent with the transition;
            - Invoke step callbacks on all hooks;
            - Handle episode completions and invoke reset callbacks;
            - Sleep to respect the timestep (if configured).
    """

    Hook = PlayerHook

    def __init__(
        self,
        environment: Environment | Environment.Factory,
        agent: Agent | Agent.Factory,
        checkpoint_path: str | Trial | None = None,
        num_steps: int | None = None,
        timestep: float | None = None,
        deterministic: bool = True,
        verbose: bool = True,
        hooks: Iterable[PlayerHook] = (),
    ):
        self.environment = environment if isinstance(environment, Environment) else environment()
        self.agent = agent if isinstance(agent, Agent) else agent.from_environment(self.environment)
        self.trial: Trial | None = None
        if checkpoint_path is not None:
            self.trial = Trial(checkpoint_path) if isinstance(checkpoint_path, str) else checkpoint_path
            checkpoint = self.trial.load_checkpoint(map_location=self.agent.device)
            self.agent.load_state_dict(checkpoint["agent"])
            self.environment.load_state_dict(checkpoint["environment"])
        self.agent.set_inference_mode(deterministic=deterministic)
        self.num_steps = num_steps
        self.timestep = self.environment.spec.timestep if timestep is None else timestep
        self.deterministic = deterministic
        self.verbose = verbose

        self.step = 0
        self.stats = EnvironmentStats(
            self.environment.num_instances,
            self.environment.spec.reward_dim,
            buffer_size=self.environment.num_instances,
        )
        self.metrics: dict[str, float] = defaultdict(float)
        self.interrupted = False

        self.hook = PlayerHookComposite()
        for hook in hooks:
            self.register_hook(hook)

    def register_hook(self, hook: Hook) -> Self:
        hook.init(self)
        self.hook.append(hook)
        return self

    def run_playing_loop(self) -> dict[str, float]:
        observation, state, _ = self.environment.reset()
        rate = utils.Rate(1 / self.timestep) if self.timestep is not None and self.timestep > 0 else None
        signal.signal(signal.SIGINT, self._sigint_handler)

        try:
            with tqdm(total=self.num_steps, disable=not self.verbose, dynamic_ncols=True) as progress_bar:
                while (self.num_steps is None or self.step < self.num_steps) and not self.interrupted:
                    action = self.agent.act(observation, state)
                    observation, state, reward, terminated, truncated, info = self.environment.step(action)
                    metrics = self.environment.get_metrics()
                    self.stats.track_step(reward)
                    for key, value in metrics.items():
                        self.metrics[key] += value

                    self.agent.step(observation, reward, terminated, truncated, state, **info)
                    self.hook.step(self.step, self.agent.transition, metrics)

                    if done_indices := get_done_indices(terminated, truncated):
                        self.stats.track_episode(done_indices)
                        if not self.environment.spec.autoreset:
                            init_observation, init_state, _ = self.environment.reset(indices=done_indices)
                            observation, state = update_observation_and_state(
                                observation, state, done_indices, init_observation, init_state
                            )
                        self.hook.reset(done_indices)

                    if rate is not None:
                        rate.tick()
                    self.step += 1
                    progress_bar.update()
        finally:
            self.hook.close()

        metrics = {
            "Mean step reward": self.stats.mean_step_reward,
            "Mean episode reward": self.stats.mean_episode_reward,
            "Mean episode length": self.stats.mean_episode_length,
        } | {key: value / self.step for key, value in self.metrics.items()}
        self._display_metrics(metrics)
        return metrics

    def _sigint_handler(self, signum, frame):
        self.interrupted = True

    def _display_metrics(self, metrics: dict[str, float]):
        formatted_metrics = {key: f"{value:.4f}" for key, value in metrics.items()}
        max_key_length = max(len(key) for key in formatted_metrics.keys())
        max_value_length = max(len(value) for value in formatted_metrics.values())
        print("┌" + "─" * (max_key_length + 2) + "┬" + "─" * (max_value_length + 2) + "┐")
        for key, value in formatted_metrics.items():
            print(f"│ {key.ljust(max_key_length)} │ {value.ljust(max_value_length)} │")
        print("└" + "─" * (max_key_length + 2) + "┴" + "─" * (max_value_length + 2) + "┘")
