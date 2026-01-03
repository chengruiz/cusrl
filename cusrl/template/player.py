import signal
from collections import defaultdict
from collections.abc import Iterable

from tqdm import tqdm

from cusrl import utils
from cusrl.template.agent import Agent
from cusrl.template.environment import Environment, get_done_indices, update_observation_and_state
from cusrl.template.trainer import EnvironmentStats
from cusrl.template.trial import Trial
from cusrl.utils.typing import Array

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

    def reset(self, indices):
        pass


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
            Path to a saved checkpoint or a Trial object. If provided, loads
            agent and environment states from the checkpoint.
        num_steps (int | None, optional):
            Maximum number of steps to execute. If None, runs indefinitely.
        timestep (float | None, optional):
            Time interval between steps in seconds. Defaults to
            `environment.spec.timestep` if not provided.
        deterministic (bool):
            Whether to run the agent in deterministic mode. If False, the agent
            will sample actions stochastically.
        verbose (bool):
            Whether to enable verbose logging.
        hooks (Iterable[PlayerHook], optional):
            A sequence of PlayerHook classes or instances to be initialized and
            called at each step and reset event.

    Methods:
        register_hook(hook: PlayerHook) -> None
            Register and initialize an additional hook to be called during play.
        run_playing_loop() -> None
            Reset environment, then repeatedly:
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
        self.stats = EnvironmentStats(
            self.environment.num_instances,
            self.environment.spec.reward_dim,
            buffer_size=self.environment.num_instances,
        )
        self.metrics: dict[str, float] = defaultdict(float)
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
        self.interrupted = False
        self.hooks = []
        self.step = 0
        for hook in hooks:
            self.register_hook(hook)

    def register_hook(self, hook: Hook):
        hook.init(self)
        self.hooks.append(hook)

    def sigint_handler(self):
        print("\033[F\033[0K\rPlaying interrupted.")
        self.interrupted = True

    def run_playing_loop(self):
        observation, state, _ = self.environment.reset()
        rate = utils.Rate(1 / self.timestep) if self.timestep is not None and self.timestep > 0 else None
        signal.signal(signal.SIGINT, lambda s, f: self.sigint_handler())
        with tqdm(total=self.num_steps, disable=not self.verbose, dynamic_ncols=True) as progress_bar:
            while (self.num_steps is None or self.step < self.num_steps) and not self.interrupted:
                action = self.agent.act(observation, state)
                observation, state, reward, terminated, truncated, info = self.environment.step(action)
                metrics = self.environment.get_metrics()
                self.stats.track_step(reward)
                self.agent.step(observation, reward, terminated, truncated, state, **info)
                for key, value in metrics.items():
                    self.metrics[key] += value
                for hook in self.hooks:
                    hook.step(self.step, self.agent.transition, metrics)
                if done_indices := get_done_indices(terminated, truncated):
                    self.stats.track_episode(done_indices)
                    if not self.environment.spec.autoreset:
                        init_observation, init_state, _ = self.environment.reset(indices=done_indices)
                        observation, state = update_observation_and_state(
                            observation, state, done_indices, init_observation, init_state
                        )
                    for hook in self.hooks:
                        hook.reset(done_indices)
                if rate is not None:
                    rate.tick()
                self.step += 1
                progress_bar.update()

        self.display_stats()

    def display_stats(self):
        metrics = {
            "Mean step reward": f"{self.stats.mean_step_reward:.4f}",
            "Mean episode reward": f"{self.stats.mean_episode_reward:.4f}",
            "Mean episode length": f"{self.stats.mean_episode_length:.4f}",
        } | {key: f"{value / self.step:.4f}" for key, value in self.metrics.items()}
        max_key_length = max(len(key) for key in metrics.keys())
        max_value_length = max(len(value) for value in metrics.values())
        print("┌" + "─" * (max_key_length + 2) + "┬" + "─" * (max_value_length + 2) + "┐")
        for key, value in metrics.items():
            print(f"│ {key.ljust(max_key_length)} │ {value.ljust(max_value_length)} │")
        print("└" + "─" * (max_key_length + 2) + "┴" + "─" * (max_value_length + 2) + "┘")
