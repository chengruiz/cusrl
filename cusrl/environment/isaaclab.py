import argparse
import importlib
from collections.abc import Sequence
from dataclasses import MISSING, dataclass, fields
from typing import Any, cast

import gymnasium as gym
import torch

import cusrl.utils
from cusrl.template import Environment
from cusrl.utils import from_dict, to_dict
from cusrl.utils.typing import Slice

__all__ = [
    "IsaacLabEnvAdapter",
    "IsaacLabEnvLauncher",
    "TrainerCfg",
    "make_isaaclab_env",
]


class IsaacLabEnvAdapter(Environment[torch.Tensor]):
    """Wraps an IsaacLab environment to conform to the cusrl.Environment
    interface."""

    def __init__(self, wrapped: gym.Env):
        from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv

        self.wrapped = wrapped
        self.unwrapped: DirectRLEnv | ManagerBasedRLEnv = wrapped.unwrapped
        self.device = self.unwrapped.device
        self.metrics = cusrl.utils.Metrics()
        super().__init__(
            num_instances=self.unwrapped.num_envs,
            observation_dim=self._get_observation_dim(),
            action_dim=self._get_action_dim(),
            state_dim=self._get_state_dim(),
            autoreset=True,
            demonstration_sampler=getattr(self.unwrapped, "collect_reference_motions", None),
            final_state_is_missing=True,
        )

        # Avoid terminal color issues
        print("\033[0m", end="")

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
            raise ValueError("Only 1D state space is supported. ")
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
        extras = cast(dict, extras)
        self.metrics.record(**extras.pop("log", {}))
        return observation, state, reward, terminated, truncated, observation_dict | extras

    def get_metrics(self):
        metrics = self.metrics.summary()
        self.metrics.clear()
        return metrics


class IsaacLabEnvLauncher(IsaacLabEnvAdapter):
    def __init__(
        self,
        id: str,
        argv: Sequence[str] | None = None,
        extensions: Sequence[str] = (),
        **kwargs: Any,
    ):
        from isaaclab.app import AppLauncher

        parser = argparse.ArgumentParser(prog="--environment-args", description="IsaacLab environment")
        parser.add_argument("--num_envs", type=int, metavar="N", help="Number of environments to simulate.")
        AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args(argv or [])
        args.device = str(cusrl.device())
        self.app_launcher = AppLauncher(args)
        self.simulation_app = self.app_launcher.app

        from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        for extension in extensions:
            importlib.import_module(extension)

        env_cfg = load_cfg_from_registry(id, "env_cfg_entry_point")
        env_cfg.sim.device = args.device
        if args.num_envs is not None:
            env_cfg.scene.num_envs = args.num_envs
        env_cfg.scene.num_envs = max(env_cfg.scene.num_envs // cusrl.utils.distributed.world_size(), 1)
        wrapped = gym.make(id, cfg=env_cfg, disable_env_checker=True, **kwargs)
        if isinstance(wrapped, DirectMARLEnv):
            wrapped = multi_agent_to_single_agent(wrapped)
        super().__init__(wrapped)

    def __del__(self):
        super().__del__()
        if hasattr(self, "simulation_app"):
            self.simulation_app.close()


def make_isaaclab_env(
    id: str,
    argv: Sequence[str] | None = None,
    play: bool = False,
    **kwargs: Any,
) -> Environment:
    if play:
        ids = id.split("-")
        ids.insert(-1, "Play")
        id = "-".join(ids)
    return IsaacLabEnvLauncher(id, argv, **kwargs)


@dataclass
class TrainerCfg:
    max_iterations: int = MISSING
    save_interval: int = MISSING
    experiment_name: str = MISSING
    agent_factory: cusrl.template.Agent.Factory = MISSING

    def __init_subclass__(cls):
        super().__init_subclass__()
        for field in fields(cls):
            if (value := getattr(cls, field.name, MISSING)) is MISSING:
                if field.default is MISSING and field.default_factory is MISSING:
                    raise ValueError(f"The default value or factory of field '{field.name}' is not defined.")
            elif field.name not in cls.__annotations__:  # will be processed by dataclass
                field.default = value
                try:
                    delattr(cls, field.name)
                except AttributeError:
                    pass

    def __post_init__(self):
        # Manually set the serialization methods to each instance
        self.to_dict = self._to_dict
        self.from_dict = self._update_from_dict

    def _to_dict(self):
        # Removing the methods temporarily to avoid recursion
        del self.to_dict
        data = to_dict(self)
        self.to_dict = self._to_dict
        return data

    def _update_from_dict(self, data):
        # Removing the methods temporarily to avoid recursion
        del self.from_dict
        updated_obj = from_dict(self, data)
        for field in fields(self):
            setattr(self, field.name, getattr(updated_obj, field.name))
        self.from_dict = self._update_from_dict
