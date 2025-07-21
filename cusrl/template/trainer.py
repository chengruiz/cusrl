import os
import pickle
import sys
from collections.abc import Callable, Iterable

import git
import numpy as np
import objprint
import torch

import cusrl
from cusrl.template.agent import Agent
from cusrl.template.environment import Environment, get_done_indices, update_observation_and_state
from cusrl.template.logger import LoggerFactoryLike
from cusrl.template.trial import Trial
from cusrl.utils import CONFIG, Timer, distributed, is_main_process
from cusrl.utils.helper import float_fmt
from cusrl.utils.typing import Slice

__all__ = ["Trainer"]


class EnvironmentStats:
    def __init__(self, num_envs: int, reward_dim: int = 1, buffer_size: int = 100):
        self.num_envs = num_envs
        self.reward_dim = reward_dim
        self.episode_rew = torch.zeros([num_envs, reward_dim])
        self.episode_len = torch.zeros([num_envs, 1])
        self.rew_buffer = torch.zeros([buffer_size, reward_dim])
        self.len_buffer = torch.zeros([buffer_size, 1])
        self.num_episodes = 0
        self.total_steps = 0

        self.num_steps = 0
        self.reward = torch.zeros([reward_dim])

    def track_step(self, reward):
        reward = torch.as_tensor(reward, device="cpu")
        self.total_steps += self.num_envs
        self.episode_rew += reward
        self.episode_len += 1

        self.reward += reward.mean(dim=0)
        self.num_steps += 1

    def clear_step_info(self):
        self.reward.fill_(0.0)
        self.num_steps = 0

    def track_episode(self, indices: Slice):
        episode_rew = self.episode_rew[indices]
        episode_len = self.episode_len[indices]
        num_episodes = episode_rew.size(0)
        index = (torch.arange(num_episodes) + self.num_episodes) % self.rew_buffer.size(0)
        self.rew_buffer[index] = episode_rew
        self.len_buffer[index] = episode_len
        self.num_episodes += num_episodes
        self.episode_rew[indices] = 0.0
        self.episode_len[indices] = 0.0

    @property
    def mean_step_reward(self) -> float | list[float]:
        mean_reward = self.reward / self.num_steps if self.num_steps else self.reward
        return mean_reward.tolist() if self.reward_dim > 1 else mean_reward.item()

    @property
    def mean_episode_reward(self) -> float | list[float]:
        if self.num_episodes == 0:
            return 0.0 if self.reward_dim == 1 else [0.0] * self.reward_dim
        mean_episode_reward = self.rew_buffer[: self.num_episodes].mean(dim=0)
        return mean_episode_reward.tolist() if self.reward_dim > 1 else mean_episode_reward.item()

    @property
    def mean_episode_length(self) -> float:
        if self.num_episodes == 0:
            return 0.0
        return self.len_buffer[: self.num_episodes].mean().item()


def save_version_info(output_dir: str, workspace: str | None = None):
    workspace_str: str = os.getcwd() if workspace is None else os.path.abspath(workspace)
    try:
        repo = git.Repo(workspace_str, search_parent_directories=True)
    except git.InvalidGitRepositoryError:
        print(f"'{workspace_str}' is not a git repository.")
        return

    repo_dir = repo.working_tree_dir
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/workspace.txt", "w") as f:
        f.write(str(repo_dir))
    with open(f"{output_dir}/git_log.txt", "w") as f:
        f.write(repo.git.log("-3"))
    with open(f"{output_dir}/git_diff.patch", "w") as f:
        f.write(repo.git.diff("HEAD"))
    with open(f"{output_dir}/git_status.txt", "w") as f:
        f.write(repo.git.status())
    with open(f"{output_dir}/version.txt", "w") as f:
        try:
            version = repo.git.describe("--tags")
        except git.GitError:
            version = "unknown"
        f.write(version)


class Trainer:
    """Orchestrates and manages a reinforcement learning training loop.

    It handles:
      - Initializing the environment, agent, logger, checkpoints, and statistics.
      - Running the main training loop for the specified number of iterations.
      - Collecting experience by alternating between agent and environment steps.
      - Updating the agent when ready, logging performance metrics, and saving checkpoints.

    Args:
        environment (Environment | Environment.Factory):
            Either an instantiated Environment or a factory that produces one.
        agent_factory (Agent.Factory):
            Factory that creates an Agent compatible with the environment.
        logger_factory (LoggerFactoryLike | None):
            Factory for a logger to persist checkpoints and metrics; active on the main process only.
        num_iterations (int):
            Total number of training iterations to execute.
        init_iteration (int | None):
            If provided, resume training from this iteration (overrides any loaded checkpoint).
        save_interval (int):
            Number of iterations between automatic checkpoints.
        checkpoint_path (str | None):
            Path to load a previous training checkpoint from.
        verbose (bool):
            Whether to print progress and checkpoint messages (only on the main process).
        callbacks (Iterable[Callable[['Trainer'], None]]):
            Sequence of functions to call after each iteration.

    Methods:
        dump_obj(obj, filename):
            Serialize an arbitrary object into the logger's info directory.
        register_callback(callback):
            Add a new callback to be executed at the end of each iteration.
        run_training_loop():
            Execute the training loop until reaching num_iterations.
    """

    def __init__(
        self,
        environment: Environment | Environment.Factory,
        agent_factory: Agent.Factory,
        logger_factory: LoggerFactoryLike | None = None,
        num_iterations: int = 1000,
        init_iteration: int | None = None,
        save_interval: int = 50,
        checkpoint_path: str | None = None,
        verbose: bool = True,
        callbacks: Iterable[Callable[["Trainer"], None]] = (),
    ):
        self.environment = environment if isinstance(environment, Environment) else environment()
        self.agent: Agent = agent_factory.from_environment(self.environment)
        self.verbose = verbose and is_main_process()
        self.iteration = self._load_checkpoint(checkpoint_path)
        if init_iteration is not None:
            self.iteration = init_iteration
        self.agent.set_iteration(self.iteration)
        self.logger = None if logger_factory is None or not is_main_process() else logger_factory()

        self.num_iterations = num_iterations
        self.save_interval = save_interval
        self.callbacks: list[Callable[[Trainer], None]] = list(callbacks)

        self.stats = EnvironmentStats(self.environment.num_instances, self.environment.spec.reward_dim)
        self.timer = Timer()
        self._save_trial_info()

    def dump_object(self, obj: object, filename: str):
        if self.logger is None or obj is None:
            return
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(f"{self.logger.info_dir}/{filename}", "wb") as f:
            if isinstance(obj, bytes):
                f.write(obj)
                return

            try:
                pickle.dump(obj, f)
            except Exception as error:
                print(f"Failed to dump object to '{filename}' due to: {error}")

    def register_callback(self, callback: Callable[["Trainer"], None]):
        self.callbacks.append(callback)

    def run_training_loop(self):
        self._save_checkpoint()

        with self.timer.record("environment"):
            observation, state, _ = self.environment.reset()
        while self.iteration < self.num_iterations:
            observation, state = self._rollout_and_update(observation, state)
            for callback in self.callbacks:
                callback(self)
            self.iteration += 1
            if self.iteration % self.save_interval == 0:
                self._save_checkpoint()

    def _rollout_and_update(self, observation, state):
        while True:
            with self.timer.record("agent"):
                action = self.agent.act(observation, state)
            with self.timer.record("environment"):
                next_observation, next_state, reward, terminated, truncated, info = self.environment.step(action)
                self.stats.track_step(reward)
            with self.timer.record("agent"):
                ready_to_update = self.agent.step(next_observation, reward, terminated, truncated, next_state, **info)
            with self.timer.record("environment"):
                if done_indices := get_done_indices(terminated, truncated):
                    if not self.environment.spec.autoreset:
                        init_observation, init_state, _ = self.environment.reset(indices=done_indices)
                        next_observation, next_state = update_observation_and_state(
                            next_observation, next_state, done_indices, init_observation, init_state
                        )
                    self.stats.track_episode(done_indices)
            observation, state = next_observation, next_state
            if ready_to_update:
                break

        with self.timer.record("agent"):
            agent_info = self.agent.update()
        self._log_info(agent_info)
        return observation, state

    def _save_checkpoint(self):
        if self.logger is None:
            return
        if self.verbose:
            print(f"Iteration {self.iteration}: Saving current checkpoint.")
        self.logger.save_checkpoint(
            {
                "agent": self.agent.state_dict(),
                "environment": self.environment.state_dict(),
                "iteration": self.iteration,
            },
            iteration=self.iteration,
        )
        if self.verbose:
            print(f"\033[F\033[0K\rIteration {self.iteration}: Checkpoint saved.")

    def _load_checkpoint(self, checkpoint_path: str | None):
        if checkpoint_path is None:
            return 0
        trial = Trial(checkpoint_path, verbose=distributed.is_main_process())
        checkpoint = trial.load_checkpoint(map_location=self.agent.device)
        self.environment.load_state_dict(checkpoint["environment"])
        self.agent.load_state_dict(checkpoint["agent"])

        if self.verbose:
            print(f"Checkpoint loaded from '{checkpoint_path}'.")
        return checkpoint["iteration"]

    def _save_trial_info(self):
        if self.logger is None:
            return
        self.logger.save_info(objprint.objstr(self.agent), "agent_info.txt")
        self.logger.save_info(" ".join(sys.orig_argv), "command.txt")
        if (seed := CONFIG.seed) is not None:
            self.logger.save_info(str(seed), "seed.txt")
        save_version_info(f"{self.logger.info_dir}/workspace")
        save_version_info(f"{self.logger.info_dir}/cusrl", cusrl.__path__[0])

    def _log_info(self, info: dict):
        info.update(self.environment.get_metrics())
        info["Metric/episode_length"] = self.stats.mean_episode_length
        info["Metric/episode_reward"] = self.stats.mean_episode_reward
        info["Metric/reward"] = self.stats.mean_step_reward
        info["Perf/environment_time"] = self.timer["environment"]
        info["Perf/agent_time"] = self.timer["agent"]
        info = distributed.average_dict(info)

        world_size = distributed.world_size()
        num_steps = self.stats.num_steps * self.environment.num_instances * world_size
        info["Perf/environment_step"] = self.stats.total_steps * world_size
        info["Perf/environment_fps"] = num_steps / info["Perf/environment_time"]
        info["Perf/agent_fps"] = num_steps / info["Perf/agent_time"]
        self.timer.clear()
        self.stats.clear_step_info()

        if self.logger is not None:
            self.logger.log(info, self.iteration)

        if not self.verbose:
            return
        episode_length_str = float_fmt(info["Metric/episode_length"], 6)
        episode_reward_str = float_fmt(np.sum(info["Metric/episode_reward"]), 6)
        step_reward_str = float_fmt(np.sum(info["Metric/reward"]), 6)
        environment_time = float_fmt(info["Perf/environment_time"], 4)
        agent_time = float_fmt(info["Perf/agent_time"], 4)
        header = f" Iteration {self.iteration + 1} / {self.num_iterations} "
        print(
            f"┌{header.center(34, '─')}┐",
            f"│ mean episode length     {episode_length_str:<8} │",
            f"│ mean episode reward     {episode_reward_str:<8} │",
            f"│ mean step reward        {step_reward_str   :<8} │",
            f"│ time consumption     {environment_time} / {agent_time} │",
            f"└{'─' * 34}┘",
            sep="\n",
        )
