from collections.abc import Iterable

from cusrl import utils
from cusrl.template.agent import Agent
from cusrl.template.environment import Environment, get_done_indices, update_observation_and_state
from cusrl.template.trial import Trial

__all__ = ["Player"]


class PlayerHook:
    player: "Player"

    def init(self, player: "Player"):
        self.player = player
        return self

    def step(self, step: int, transition: dict, metrics: dict):
        pass

    def reset(self, indices):
        pass


class Player:
    """Orchestrates a playing loop between an Agent and an Environment,
    also manages initialization, checkpoint loading, stepping, and hook callbacks.

    Args:
        environment (Environment | Environment.Factory):
            An Environment instance or a factory that produces one.
        agent (Agent | Agent.Factory):
            An Agent instance or a factory that produces one for the given environment.
        checkpoint_path (str | Trial | None, optional):
            Path to a saved checkpoint or a Trial object. If provided, loads agent and
            environment states from the checkpoint.
        num_steps (int | None, optional):
            Maximum number of steps to execute. If None, runs indefinitely.
        timestep (float | None, optional):
            Time interval between steps in seconds. Defaults to
            `environment.spec.timestep` if not provided.
        deterministic (bool):
            Whether to run the agent in deterministic mode. If False, the agent will
            sample actions stochastically.
        verbose (bool):
            Whether to enable verbose logging.
        hooks (Iterable[PlayerHook], optional):
            A sequence of PlayerHook classes or instances to be initialized and called
            at each step and reset event.

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
        if checkpoint_path is not None:
            trial = Trial(checkpoint_path) if isinstance(checkpoint_path, str) else checkpoint_path
            checkpoint = trial.load_checkpoint(map_location=self.agent.device)
            self.agent.load_state_dict(checkpoint["agent"])
            self.environment.load_state_dict(checkpoint["environment"])
        self.agent.set_inference_mode(deterministic=deterministic)
        self.num_steps = num_steps
        self.timestep = self.environment.spec.timestep if timestep is None else timestep
        self.deterministic = deterministic
        self.verbose = verbose
        self.hooks = [hook.init(self) for hook in hooks]

    def register_hook(self, hook: Hook):
        self.hooks.append(hook.init(self))

    def run_playing_loop(self):
        observation, state, _ = self.environment.reset()
        rate = utils.Rate(1 / self.timestep) if self.timestep is not None else None
        step = 0
        while self.num_steps is None or step < self.num_steps:
            action = self.agent.act(observation, state)
            observation, state, reward, terminated, truncated, info = self.environment.step(action)
            self.agent.step(observation, reward, terminated, truncated, state, **info)
            for hook in self.hooks:
                hook.step(step, self.agent.transition, self.environment.get_metrics())
            if done_indices := get_done_indices(terminated, truncated):
                if not self.environment.spec.autoreset:
                    init_observation, init_state, _ = self.environment.reset(indices=done_indices)
                    observation, state = update_observation_and_state(
                        observation, state, done_indices, init_observation, init_state
                    )
                for hook in self.hooks:
                    hook.reset(done_indices)
            if rate is not None:
                rate.tick()
            step += 1
