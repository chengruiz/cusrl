import signal
from collections.abc import Iterable

import torch
from tqdm import tqdm
from typing_extensions import Self

from cusrl import utils
from cusrl.template.agent import Agent, AgentFactory
from cusrl.template.environment import (
    Environment,
    EnvironmentFactoryLike,
    get_done_indices,
    update_observation_and_state,
)
from cusrl.template.trainer import EnvironmentStats
from cusrl.template.trial import Trial
from cusrl.utils.typing import Array, Slice

__all__ = ["Player"]


class PlayerHook:
    """Base class for hooks that receive callbacks during the playing loop.

    Subclass this to observe or react to play-time events such as steps, episode
    resets, and loop completion. All callback methods are no-ops by default so
    subclasses only need to override the events they care about.

    Attributes:
        player (Player): The owning player instance, set by :meth:`init`.
        agent (Agent): Shortcut to ``player.agent``.
        environment (Environment): Shortcut to ``player.environment``.
    """

    player: "Player"
    agent: Agent
    environment: Environment

    def init(self, player: "Player"):
        """Called once when the hook is registered with a :class:`Player`.

        Stores references to the player, agent, and environment for use in
        subsequent callbacks.

        Args:
            player: The player instance that owns this hook.
        """
        self.player = player
        self.agent = player.agent
        self.environment = player.environment

    def step(self, step: int, transition: dict[str, Array]):
        """Called after every environment step.

        Args:
            step: The zero-based step index within the current playing loop.
            transition: The agent's latest transition dictionary.
        """

    def reset(self, indices: Slice):
        """Called when one or more environment instances finish an episode.

        Args:
            indices: Indices of the environment instances that were reset.
        """

    def close(self):
        """Called once when the playing loop ends."""


class PlayerHookComposite(PlayerHook, list[PlayerHook]):
    """Delegates every callback to all contained hooks in order."""

    def init(self, player: "Player"):
        for hook in self:
            hook.init(player)

    def step(self, step: int, transition):
        for hook in self:
            hook.step(step, transition)

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
        environment (Environment | EnvironmentFactoryLike):
            An Environment instance or a factory that produces one.
        agent (Agent | AgentFactory):
            An Agent instance or a factory that produces one for the given
            environment.
        checkpoint_path (str | Trial | None, optional):
            Path to a saved checkpoint or a :class:`Trial` object. If provided,
            loads the states of the agent and the environment states from the
            checkpoint. Defaults to ``None`` (no checkpoint loading).
        num_steps (int | None, optional):
            Maximum number of environment steps to execute across all instances.
            If ``None``, the step count does not limit the loop. Defaults to
            ``None``.
        num_episodes (int | None, optional):
            Minimum number of episodes each environment instance must complete
            before the loop ends. If ``None``, the episode count does not limit
            the loop. Defaults to ``None``.
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

    Example::

        player = Player(
            environment=env_factory,
            agent=agent_factory,
            checkpoint_path="logs/trial_01",
            num_episodes=10,
            hooks=[MyRecordingHook()],
        )
        metrics = player.run_playing_loop()
    """

    Hook = PlayerHook

    def __init__(
        self,
        environment: Environment | EnvironmentFactoryLike,
        agent: Agent | AgentFactory,
        checkpoint_path: str | Trial | None = None,
        num_steps: int | None = None,
        num_episodes: int | None = None,
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
        self.num_episodes = num_episodes
        self.timestep = self.environment.spec.timestep if timestep is None else timestep
        self.deterministic = deterministic
        self.verbose = verbose

        self.step_count = 0
        self.stats = EnvironmentStats(
            self.environment.num_instances,
            self.environment.spec.reward_dim,
            buffer_size=self.environment.num_instances,
        )
        self.episode_count = torch.zeros(self.environment.num_instances, dtype=torch.long)
        self.interrupted = False

        self.hook = PlayerHookComposite()
        for hook in hooks:
            self.register_hook(hook)

    def register_hook(self, hook: PlayerHook) -> Self:
        """Register and initialize a hook to receive callbacks during play.

        Args:
            hook: The hook instance to register.

        Returns:
            This player instance, allowing method chaining.
        """
        hook.init(self)
        self.hook.append(hook)
        return self

    def run_playing_loop(self) -> dict[str, float]:
        """Execute the main playing loop.

        Resets the environment, then repeatedly queries the agent for actions,
        steps the environment, invokes hook callbacks, and handles episode
        completions. The loop terminates when ``num_steps`` is exhausted,
        every instance has completed ``num_episodes`` episodes, or the process
        receives *SIGINT*. If both limits are ``None``, the loop runs until
        interrupted.

        Returns:
            A dictionary of aggregated metrics (mean rewards, episode length,
            and any environment-specific metrics).
        """
        observation, state, _ = self.environment.reset()
        rate = utils.Rate(1 / self.timestep) if self.timestep is not None and self.timestep > 0 else None

        try:
            prev_handler = signal.signal(signal.SIGINT, self._sigint_handler)
        except ValueError:
            prev_handler = None

        try:
            with tqdm(total=self.num_steps, disable=not self.verbose, dynamic_ncols=True) as progress_bar:
                while (self.num_steps is None or self.step_count < self.num_steps) and not self.interrupted:
                    action = self.agent.act(observation, state)
                    observation, state, reward, terminated, truncated, info = self.environment.step(action)
                    self.agent.step(observation, reward, terminated, truncated, state, **info)
                    self._step_event(observation, state, reward, terminated, truncated, info)

                    if done_indices := get_done_indices(terminated, truncated):
                        if not self.environment.spec.autoreset:
                            init_observation, init_state, _ = self.environment.reset(indices=done_indices)
                            observation, state = update_observation_and_state(
                                observation, state, done_indices, init_observation, init_state
                            )
                        self._reset_event(done_indices)

                    if rate is not None:
                        rate.tick()
                    progress_bar.update()

                    if self.num_episodes is not None and (self.episode_count >= self.num_episodes).all():
                        break
        finally:
            self.hook.close()
            self.environment.close()
            if prev_handler is not None:
                signal.signal(signal.SIGINT, prev_handler)

        metrics = self._get_metrics_report()
        self._display_metrics(metrics)
        return metrics

    def _sigint_handler(self, signum, frame):
        self.interrupted = True
        # Restore default so a second Ctrl+C kills immediately.
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    def _step_event(self, observation, state, reward, terminated, truncated, info):
        self.stats.track_step(reward)
        self.hook.step(self.step_count, self.agent.transition)
        self.step_count += 1

    def _reset_event(self, done_indices: list[int]):
        self.episode_count[done_indices] += 1
        self.stats.track_episode(done_indices)
        self.hook.reset(done_indices)

    def _get_metrics_report(self) -> dict[str, float]:
        return {
            "Mean step reward": self.stats.mean_step_reward,
            "Mean episode reward": self.stats.mean_episode_reward,
            "Mean episode length": self.stats.mean_episode_length,
        } | self.environment.get_metrics()

    def _display_metrics(self, metrics: dict[str, float]):
        if not metrics:
            return

        formatted_metrics = {key: f"{value: .6g}" for key, value in metrics.items()}
        max_key_length = max(len(key) for key in formatted_metrics.keys())
        max_value_length = max(len(value) for value in formatted_metrics.values())
        print("┌" + "─" * (max_key_length + 2) + "┬" + "─" * (max_value_length + 2) + "┐")
        for key, value in formatted_metrics.items():
            print(f"│ {key.ljust(max_key_length)} │ {value.ljust(max_value_length)} │")
        print("└" + "─" * (max_key_length + 2) + "┴" + "─" * (max_value_length + 2) + "┘")
