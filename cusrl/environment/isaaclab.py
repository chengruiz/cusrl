import argparse
from collections.abc import Sequence
from typing import Any, cast

import gymnasium as gym
import torch

import cusrl.utils
from cusrl.template import Environment, EnvironmentSpec
from cusrl.utils.typing import Array, Slice

__all__ = ["IsaacLabEnvAdapter", "make_isaaclab_env"]


class IsaacLabEnvAdapter(Environment):
    def __init__(self, id: str, argv: Sequence[str] | None = None, **kwargs):
        from isaaclab.app import AppLauncher

        parser = argparse.ArgumentParser(prog="--environment-args", description="IsaacLab environment")
        parser.add_argument("--num_envs", type=int, metavar="N", help="Number of environments to simulate.")
        AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args(argv or [])
        args.device = str(cusrl.device())
        self.app_launcher = AppLauncher(args)
        self.simulation_app = self.app_launcher.app

        from isaaclab.envs import DirectMARLEnv, DirectRLEnv, ManagerBasedRLEnv, multi_agent_to_single_agent
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        env_cfg = load_cfg_from_registry(id, "env_cfg_entry_point")
        env_cfg.sim.device = args.device
        if args.num_envs is not None:
            env_cfg.scene.num_envs = args.num_envs
        env_cfg.scene.num_envs = max(env_cfg.scene.num_envs // cusrl.utils.distributed.world_size(), 1)
        isaaclab_env = gym.make(id, cfg=env_cfg, disable_env_checker=True, **kwargs)
        if isinstance(isaaclab_env.unwrapped, DirectMARLEnv):
            isaaclab_env = multi_agent_to_single_agent(isaaclab_env)
        self.wrapped: ManagerBasedRLEnv | DirectRLEnv = isaaclab_env.unwrapped
        self.device = self.wrapped.device
        self.metrics = cusrl.utils.Metrics()
        super().__init__(
            self.wrapped.num_envs,
            self._get_observation_dim(),
            self._get_action_dim(),
            self._get_state_dim(),
            EnvironmentSpec(autoreset=True, final_state_is_missing=True),
        )

        # Avoid terminal color issues
        print("\033[0m", end="")

    def __del__(self):
        if hasattr(self, "wrapped"):
            self.wrapped.close()
        if hasattr(self, "simulation_app"):
            self.simulation_app.close()

    def _get_observation_dim(self) -> int:
        if hasattr(self.wrapped, "observation_manager"):
            shape = self.wrapped.observation_manager.group_obs_dim["policy"]
        else:
            shape = self.wrapped.single_observation_space["policy"].shape

        if not len(shape) == 1:
            raise ValueError("Only 1D observation space is supported. ")
        return shape[0]

    def _get_action_dim(self) -> int:
        if hasattr(self.wrapped, "action_manager"):
            return self.wrapped.action_manager.total_action_dim
        space = self.wrapped.single_action_space
        if not len(space.shape) == 1:
            raise ValueError("Only 1D action space is supported. ")
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
            raise ValueError("Only 1D state space is supported. ")
        return shape[0]

    def reset(self, *, indices: Array | Slice | None = None):
        if indices is None:
            observation_dict, _ = self.wrapped.reset()
            self.wrapped.episode_length_buf.random_(int(self.wrapped.max_episode_length))
            observation = observation_dict.pop("policy")
            state = observation_dict.pop("critic", None)
            extras = observation_dict
        else:
            if isinstance(indices, slice):
                indices = torch.arange(self.num_instances, device=self.device)[indices]
            observation_dict, _ = self.wrapped.reset(env_ids=torch.as_tensor(indices, device=self.device))

            observation = observation_dict.pop("policy", None)
            state = observation_dict.pop("critic", None)
            extras = {key: value[indices] for key, value in observation_dict.items()}
            if observation is not None:
                observation = observation[indices]
            if state is not None:
                state = state[indices]

        return observation, state, extras

    def step(self, action):
        observation_dict, reward, terminated, truncated, extras = self.wrapped.step(action)
        observation = observation_dict.pop("policy")
        state = observation_dict.pop("critic", None)
        reward = cast(torch.Tensor, reward).unsqueeze(-1)
        terminated = cast(torch.Tensor, terminated).unsqueeze(-1)
        truncated = cast(torch.Tensor, truncated).unsqueeze(-1)
        extras = cast(dict, extras)
        self.metrics.record(**extras.pop("log", {}))
        return observation, state, reward, terminated, truncated, observation_dict | extras

    def get_metrics(self):
        metrics = self.metrics.summary()
        self.metrics.clear()
        return metrics


def make_isaaclab_env(id: str, argv: Sequence[str] | None = None, **kwargs: Any) -> Environment:
    return IsaacLabEnvAdapter(id, argv, **kwargs)
