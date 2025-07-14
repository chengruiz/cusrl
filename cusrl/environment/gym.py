import random
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import EnvSpec

import cusrl.utils
from cusrl.template import Environment
from cusrl.utils.typing import Array, Slice

__all__ = ["GymEnvAdapter", "GymVectorEnvAdapter", "make_gym_env", "make_gym_vec"]


class GymEnvAdapter(Environment):
    def __init__(self, wrapped: gym.Env):
        if not isinstance(wrapped.observation_space, gym.spaces.Box):
            raise ValueError("Only Box observation space is supported.")
        if not len(wrapped.observation_space.shape) == 1:
            raise ValueError("Only 1D observation space is supported.")
        if isinstance(wrapped.action_space, gym.spaces.Box):
            if not len(wrapped.action_space.shape) == 1:
                raise ValueError("For Box action space, only 1D action space is supported.")
            action_dim = wrapped.action_space.shape[0]
        elif isinstance(wrapped.action_space, gym.spaces.Discrete):
            action_dim = int(wrapped.action_space.n)
        else:
            raise ValueError(f"Unsupported action space type: {wrapped.action_space}.")

        super().__init__(
            num_instances=1,
            observation_dim=wrapped.observation_space.shape[0],
            action_dim=action_dim,
            spec={"gym_spec": wrapped.spec, "gym_metadata": wrapped.metadata},
        )
        wrapped.reset(seed=random.getrandbits(4))
        self.wrapped = wrapped

    def reset(self, *, indices: Array | Slice | None = None):
        observation, info = self.wrapped.reset()
        observation = observation.reshape(1, -1)
        if self.wrapped.render_mode is not None:
            self.wrapped.render()
        # TODO: process arrays in info
        return observation, None, info

    def step(self, action: Array):
        if isinstance(self.wrapped.action_space, gym.spaces.Discrete):
            action = np.argmax(action, axis=-1)
        action = action.squeeze(0)
        observation, reward, terminated, truncated, info = self.wrapped.step(action)
        observation = observation.reshape(1, -1)
        reward = np.array([[reward]], dtype=np.float32)
        terminated = np.array([[terminated]])
        truncated = np.array([[truncated]])
        if self.wrapped.render_mode is not None:
            self.wrapped.render()
        # TODO: process arrays in info
        return observation, None, reward, terminated, truncated, info


class GymVectorEnvAdapter(Environment):
    def __init__(self, wrapped: gym.vector.VectorEnv):
        if not isinstance(wrapped.single_observation_space, gym.spaces.Box):
            raise ValueError("Only Box observation space is supported.")
        if not len(wrapped.single_observation_space.shape) == 1:
            raise ValueError("Only 1D observation space is supported.")
        if isinstance(wrapped.single_action_space, gym.spaces.Box):
            if not len(wrapped.single_action_space.shape) == 1:
                raise ValueError("For Box action space, only 1D action space is supported.")
            action_dim = wrapped.single_action_space.shape[0]
        elif isinstance(wrapped.single_action_space, gym.spaces.Discrete):
            action_dim = int(wrapped.single_action_space.n)
        else:
            raise ValueError(f"Unsupported action space type: {wrapped.single_action_space}.")

        if (autoreset_mode := wrapped.metadata.get("autoreset_mode")) is None:
            if cusrl.utils.is_main_process():
                warnings.warn("GymVectorEnvAdapter: make sure 'autoreset_mode' is 'DISABLED'.")
        elif autoreset_mode != gym.vector.AutoresetMode.DISABLED:
            raise ValueError("'autoreset_mode' of vector environments must be 'DISABLED'.")

        super().__init__(
            num_instances=wrapped.num_envs,
            observation_dim=wrapped.single_observation_space.shape[0],
            action_dim=action_dim,
            spec={"gym_spec": wrapped.spec, "gym_metadata": wrapped.metadata},
        )
        wrapped.reset(seed=random.getrandbits(4))
        self.wrapped = wrapped

    def reset(self, *, indices: Array | Slice | None = None):
        if indices is None:
            observation, info = self.wrapped.reset()
            return observation, None, info
        if not isinstance(indices, np.ndarray):
            mask = np.zeros(self.num_instances, dtype=bool)
            mask[indices] = True
            indices = mask

        observation, info = self.wrapped.reset(options={"reset_mask": indices})
        if self.wrapped.render_mode is not None:
            self.wrapped.render()
        # TODO: process arrays in info
        return observation, None, info

    def step(self, action: Array):
        if isinstance(self.wrapped.single_action_space, gym.spaces.Discrete):
            action = np.argmax(action, axis=-1)
        observation, reward, terminated, truncated, info = self.wrapped.step(action)
        if isinstance(reward, np.ndarray):
            reward = reward.astype(np.float32)
        reward = reward.reshape(-1, 1)
        terminated = terminated.reshape(-1, 1)
        truncated = truncated.reshape(-1, 1)
        if self.wrapped.render_mode is not None:
            self.wrapped.render()
        # TODO: process arrays in info
        return observation, None, reward, terminated, truncated, info


def make_gym_env(
    id: str | EnvSpec,
    max_episode_steps: int | None = None,
    disable_env_checker: bool | None = None,
    **kwargs: Any,
) -> Environment:
    return GymEnvAdapter(
        gym.make(
            id=id,
            max_episode_steps=max_episode_steps,
            disable_env_checker=disable_env_checker,
            **kwargs,
        )
    )


def make_gym_vec(
    id: str | EnvSpec,
    num_envs: int = 1,
    vectorization_mode: gym.VectorizeMode | str | None = None,
    vector_kwargs: dict[str, Any] | None = None,
    wrappers: Sequence[Callable[[gym.Env], gym.Wrapper]] | None = None,
    **kwargs,
) -> Environment:
    return GymVectorEnvAdapter(
        gym.make_vec(
            id=id,
            num_envs=num_envs,
            vectorization_mode=vectorization_mode,
            vector_kwargs=(vector_kwargs or {}) | {"autoreset_mode": gym.vector.AutoresetMode.DISABLED},
            wrappers=wrappers,
            **kwargs,
        )
    )
