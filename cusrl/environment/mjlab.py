from typing import cast

import gymnasium as gym
import torch

import cusrl.utils
from cusrl.template import Environment
from cusrl.utils.typing import Slice

__all__ = [
    "MjlabEnvAdapter",
]


class MjlabEnvAdapter(Environment[torch.Tensor]):
    """Wraps an mjlab environment to conform to the cusrl.Environment
    interface."""

    def __init__(self, wrapped: gym.Env):
        from mjlab.envs import ManagerBasedRlEnv

        self.wrapped = wrapped
        self.unwrapped: ManagerBasedRlEnv = wrapped.unwrapped
        self.device = torch.device(self.unwrapped.device)
        self.metrics = cusrl.utils.Metrics()
        super().__init__(
            num_instances=self.unwrapped.num_envs,
            observation_dim=self._get_observation_dim(),
            action_dim=self._get_action_dim(),
            state_dim=self._get_state_dim(),
            autoreset=True,
            final_state_is_missing=True,
        )

    def __del__(self):
        if hasattr(self, "wrapped"):
            self.wrapped.close()

    def _get_observation_dim(self) -> int:
        if hasattr(self.unwrapped, "observation_manager"):
            shape = self.unwrapped.observation_manager.group_obs_dim["policy"]
        else:
            shape = self.unwrapped.single_observation_space["policy"].shape

        if not len(shape) == 1:
            raise ValueError("Only 1D observation space is supported. ")
        return shape[0]

    def _get_action_dim(self) -> int:
        if hasattr(self.unwrapped, "action_manager"):
            return self.unwrapped.action_manager.total_action_dim
        space = self.unwrapped.single_action_space
        if not len(space.shape) == 1:
            raise ValueError("Only 1D action space is supported. ")
        return space.shape[0]

    def _get_state_dim(self) -> int | None:
        shape = None
        if hasattr(self.unwrapped, "observation_manager"):
            shape = self.unwrapped.observation_manager.group_obs_dim.get("critic")
        else:
            space = self.unwrapped.single_observation_space.get("critic")
            if space is not None:
                shape = space.shape

        if shape is None:
            return None
        if not len(shape) == 1:
            raise ValueError("Only 1D state space is supported.")
        return shape[0]

    def reset(self, *, indices: torch.Tensor | Slice | None = None):
        if indices is None:
            observation_dict, _ = self.wrapped.reset()
            self.unwrapped.episode_length_buf.random_(int(self.unwrapped.max_episode_length))
            observation = observation_dict.pop("policy")
            state = observation_dict.pop("critic", None)
            extras = observation_dict
        else:
            if isinstance(indices, slice):
                indices = torch.arange(self.num_instances, device=self.device)[indices]
            observation_dict, _ = self.unwrapped.reset(env_ids=torch.as_tensor(indices, device=self.device))

            observation = observation_dict.pop("policy", None)
            state = observation_dict.pop("critic", None)
            extras = {key: value[indices] for key, value in observation_dict.items()}
            if observation is not None:
                observation = observation[indices]
            if state is not None:
                state = state[indices]

        return observation, state, extras

    def step(self, action: torch.Tensor):
        observation_dict, reward, terminated, truncated, extras = self.wrapped.step(action)
        observation = observation_dict.pop("policy")
        state = observation_dict.pop("critic", None)
        reward = cast(torch.Tensor, reward).unsqueeze(-1)
        terminated = cast(torch.Tensor, terminated).unsqueeze(-1)
        truncated = cast(torch.Tensor, truncated).unsqueeze(-1)
        extras = cast(dict, extras).copy()
        self.metrics.record(**extras.pop("log", {}))
        return observation, state, reward, terminated, truncated, observation_dict | extras

    def get_metrics(self):
        metrics = self.metrics.summary()
        self.metrics.clear()
        return metrics
