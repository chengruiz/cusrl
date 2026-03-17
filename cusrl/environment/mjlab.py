from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch

import cusrl.utils
from cusrl.template import Agent, Environment, Player, Trial
from cusrl.template.environment import get_done_indices
from cusrl.utils.typing import Slice

__all__ = [
    "MjlabEnvAdapter",
    "MjlabPlayer",
    "make_mjlab_env",
]


class MjlabEnvAdapter(Environment[torch.Tensor]):
    """Wraps an mjlab environment to conform to the cusrl.Environment
    interface."""

    def __init__(self, wrapped):
        from mjlab.envs import ManagerBasedRlEnv

        self.wrapped: ManagerBasedRlEnv = cast(ManagerBasedRlEnv, wrapped)
        self.device = torch.device(self.wrapped.device)
        self.metrics = cusrl.utils.Metrics()
        super().__init__(
            num_instances=self.wrapped.num_envs,
            observation_dim=self._get_observation_dim(),
            action_dim=self._get_action_dim(),
            state_dim=self._get_state_dim(),
            autoreset=True,
            final_state_is_missing=True,
        )

    def close(self):
        if hasattr(self, "wrapped"):
            self.wrapped.close()

    def _get_observation_dim(self) -> int:
        if hasattr(self.wrapped, "observation_manager"):
            shape = self.wrapped.observation_manager.group_obs_dim["actor"]
        else:
            shape = self.wrapped.single_observation_space["actor"].shape

        if not len(shape) == 1:
            raise ValueError("Only 1D observation spaces are supported")
        return shape[0]

    def _get_action_dim(self) -> int:
        if hasattr(self.wrapped, "action_manager"):
            return self.wrapped.action_manager.total_action_dim
        space = self.wrapped.single_action_space
        if not len(space.shape) == 1:
            raise ValueError("Only 1D action spaces are supported")
        return space.shape[0]

    def _get_state_dim(self) -> int | None:
        shape = None
        if hasattr(self.wrapped, "observation_manager"):
            shape = self.wrapped.observation_manager.group_obs_dim.get("critic")
        else:
            space = self.wrapped.single_observation_space.get("critic")
            if space is not None:
                shape = space.shape

        if shape is None:
            return None
        if not len(shape) == 1:
            raise ValueError("Only 1D state spaces are supported")
        return shape[0]

    def reset(
        self,
        *,
        indices: torch.Tensor | Slice | None = None,
        randomize_episode_progress: bool = False,
    ):
        if indices is None:
            observation_dict, _ = self.wrapped.reset()
            observation = observation_dict.pop("actor")
            state = observation_dict.pop("critic", None)
            extras = observation_dict
        else:
            if isinstance(indices, slice):
                indices = torch.arange(self.num_instances, device=self.device)[indices]
            observation_dict, _ = self.wrapped.reset(env_ids=torch.as_tensor(indices, device=self.device))

            observation = observation_dict.pop("actor", None)
            state = observation_dict.pop("critic", None)
            extras = {key: value[indices] for key, value in observation_dict.items()}
            if observation is not None:
                observation = observation[indices]
            if state is not None:
                state = state[indices]

        if randomize_episode_progress:
            self.wrapped.episode_length_buf[indices] = torch.randint_like(
                self.wrapped.episode_length_buf[indices], int(self.wrapped.max_episode_length)
            )

        return observation, state, extras

    def step(self, action: torch.Tensor):
        observation_dict, reward, terminated, truncated, extras = self.wrapped.step(action)
        observation = observation_dict.pop("actor")
        state = observation_dict.pop("critic", None)
        reward = cast(torch.Tensor, reward).unsqueeze(-1)
        terminated = cast(torch.Tensor, terminated).unsqueeze(-1)
        truncated = cast(torch.Tensor, truncated).unsqueeze(-1)
        extras = cast(dict, extras).copy()

        log = extras.pop("log", {}).copy()
        num_finished_episodes = (terminated | truncated).sum().cpu().item()
        for key, value in tuple(log.items()):
            if key.startswith("Episode_Reward/"):
                log[key] = value.repeat(num_finished_episodes)
            elif key.startswith("Episode_Termination/"):
                termination_flags = torch.zeros(num_finished_episodes)
                termination_flags[:value] = 1.0
                log[key] = termination_flags
        self.metrics.update(log)
        return observation, state, reward, terminated, truncated, observation_dict | extras

    def get_metrics(self):
        metrics = self.metrics.summary()
        self.metrics.clear()
        return metrics


class MjlabPlayer(Player):
    """A Player implementation for playing with mjlab environments using the
    mjlab built-in viewers."""

    def __init__(
        self,
        environment: Environment | Environment.Factory,
        agent: Agent | Agent.Factory,
        checkpoint_path: str | Trial | None = None,
        num_steps: int | None = None,
        num_episodes: int | None = None,
        timestep: float | None = None,
        deterministic: bool = True,
        verbose: bool = True,
        hooks: Iterable[Player.Hook] = (),
    ):
        super().__init__(
            environment=environment,
            agent=agent,
            checkpoint_path=checkpoint_path,
            num_steps=num_steps,
            num_episodes=num_episodes,
            timestep=timestep,
            deterministic=deterministic,
            verbose=verbose,
            hooks=hooks,
        )
        self.is_first_step = True

    def run_playing_loop(self) -> dict[str, float]:
        from mjlab.rl import RslRlVecEnvWrapper
        from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

        environment = cast(MjlabEnvAdapter, self.environment)
        cfg = environment.wrapped.cfg
        if cfg.headless or cfg.viewer_type is None:
            return super().run_playing_loop()

        native_environment = RslRlVecEnvWrapper(environment.wrapped)
        if cfg.viewer_type == "native":
            viewer = NativeMujocoViewer(native_environment, self)
        elif cfg.viewer_type == "viser":
            viewer = ViserPlayViewer(native_environment, self)
        else:
            raise ValueError(f"Unsupported viewer type '{cfg.viewer_type}'")
        viewer.run(self.num_steps)
        metrics = self._get_metrics_report()
        self._display_metrics(metrics)
        return metrics

    def __call__(self, observation_dict):
        """Acts like an Mjlab policy."""
        observation = observation_dict.pop("actor")
        state = observation_dict.pop("critic", None)

        if not self.is_first_step:
            reward = self.environment.wrapped.reward_buf.clone().unsqueeze(-1)
            terminated = self.environment.wrapped.termination_manager.terminated.clone().unsqueeze(-1)
            truncated = self.environment.wrapped.termination_manager.time_outs.clone().unsqueeze(-1)
            extras = cast(dict, self.environment.wrapped.extras).copy()
            self.environment.metrics.record(**extras.pop("log", {}))
            self.agent.step(
                next_observation=observation,
                next_state=state,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                **observation_dict,
            )
            self._step_event(observation, state, reward, terminated, truncated, observation_dict)

            if done_indices := get_done_indices(terminated, truncated):
                self._reset_event(done_indices)
        else:
            self.is_first_step = False

        action = self.agent.act(observation, state)
        return action


def make_mjlab_env(
    id: str,
    argv: Sequence[str] = (),
    play: bool = False,
    device: str | torch.device | None = None,
    **kwargs: Any,
) -> MjlabEnvAdapter:
    import mjlab
    import tyro
    from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
    from mjlab.tasks.registry import load_env_cfg

    @dataclass
    class ManagerBasedRlEnvPlayCfg(ManagerBasedRlEnvCfg):
        headless: bool = False
        viewer_type: Literal[None, "native", "viser"] = "viser"

    config_class = ManagerBasedRlEnvPlayCfg if play else ManagerBasedRlEnvCfg
    env_cfg = load_env_cfg(id, play=play)
    env_cfg = tyro.cli(
        config_class,
        args=argv,
        default=env_cfg,
        config=mjlab.TYRO_FLAGS,
    )

    env = ManagerBasedRlEnv(env_cfg, device=str(cusrl.utils.device(device)), **kwargs)
    return MjlabEnvAdapter(env)
