from collections.abc import Sequence

import numpy as np
import torch

from cusrl.hook.symmetry import SymmetryDef
from cusrl.template import ActorCritic, Hook
from cusrl.utils import RunningMeanStd, mean_var_count
from cusrl.utils.export import ExportGraph
from cusrl.utils.typing import Slice

__all__ = ["ObservationNormalization"]


class ObservationNormalization(Hook[ActorCritic]):
    """Normalizes observations and states using a running mean and standard deviation.

    This hook maintains a running estimate of the mean and standard deviation for
    observations and, if available, states. It intercepts transitions during data
    collection to normalize the `observation`, `state`, `next_observation`, and
    `next_state` fields. The original, unnormalized values are preserved under
    keys with an "original_" prefix.

    The running statistics are updated with new data from each step, unless the
    agent is in inference mode or the `frozen` attribute is set to `True`.

    The hook also handles scenarios where the observation is a subset of the
    state or where there is symmetry in the observations or states. It correctly
    synchronizes statistics across distributed processes. During model export,
    it attaches forward pre-hooks to the actor and critic models to ensure that
    inputs are automatically normalized.

    Args:
        max_count:
            The maximum count for the running statistics to prevent numerical
            overflow. Defaults to None.
        defer_synchronization:
            If True, synchronization of running statistics in a distributed setting
            is deferred until the end of a rollout. This can improve performance
            by reducing the frequency of synchronization. Defaults to False.
    """

    observation_rms: RunningMeanStd
    state_rms: RunningMeanStd | None = None
    _mirror_observation: SymmetryDef | None = None
    _mirror_state: SymmetryDef | None = None
    _observation_is_subset_of_state: Slice | torch.Tensor | None = None

    MODULES = ["observation_rms", "state_rms"]
    MUTABLE_ATTRS = ["frozen"]

    def __init__(self, max_count: int | None = None, defer_synchronization: bool = False):
        if max_count is not None and max_count <= 0:
            raise ValueError("'max_count' must be positive or None.")
        self.max_count = max_count
        self.frozen: bool = False
        self.defer_synchronization = defer_synchronization
        self._last_done: torch.Tensor | None = None

    def init(self):
        # Retrieve and normalize the subset index spec
        env_spec = self.agent.environment_spec
        observation_is_subset_of_state = env_spec.observation_is_subset_of_state
        if observation_is_subset_of_state is not None:
            if not self.agent.has_state:
                raise ValueError("'observation_is_subset_of_state' is set but state is not defined.")
            # Convert numpy or list indices to a tensor for consistent indexing
            if isinstance(observation_is_subset_of_state, (np.ndarray, Sequence)):
                observation_is_subset_of_state = self.agent.to_tensor(np.asarray(observation_is_subset_of_state))
        self._observation_is_subset_of_state = observation_is_subset_of_state

        if self._observation_is_subset_of_state is not None:
            self.observation_rms = self.__make_rms(self.agent.observation_dim)
        else:
            self.observation_rms = self.__make_rms(
                self.agent.observation_dim, self.max_count, env_spec.observation_stat_groups
            )
        if self.agent.has_state:
            self.state_rms = self.__make_rms(self.agent.state_dim, self.max_count, env_spec.state_stat_groups)
        self._mirror_observation = env_spec.mirror_observation
        self._mirror_state = env_spec.mirror_state

    def pre_act(self, transition: dict):
        observation, state = transition["observation"], transition.get("state")
        if self._last_done is None or not self.agent.environment_spec.final_state_is_missing:
            self.__update_rms(observation, state, self._last_done)

        transition["original_observation"] = observation
        transition["observation"] = self.observation_rms.normalize(observation)
        if state is not None:
            transition["original_state"] = state
            transition["state"] = self.state_rms.normalize(state)

    def post_step(self, transition: dict):
        next_observation, next_state = transition["next_observation"], transition.get("next_state")
        self.__update_rms(next_observation, next_state)
        self._last_done = transition["done"].squeeze(-1)

        transition["original_next_observation"] = next_observation
        transition["next_observation"] = self.observation_rms.normalize(next_observation)
        if next_state is not None:
            transition["original_next_state"] = next_state
            transition["next_state"] = self.state_rms.normalize(next_state)

    def __make_rms(
        self,
        num_channels: int,
        max_count: int | None = None,
        stat_groups: tuple[tuple[int, int], ...] = (),
    ):
        normalizer = RunningMeanStd(num_channels, max_count=max_count).to(self.agent.device)
        for group in stat_groups:
            normalizer.register_stat_group(*group)
        return normalizer

    def __update_rms(
        self,
        observation: torch.Tensor,
        state: torch.Tensor | None,
        indices: torch.Tensor | None = None,
    ):
        if self.agent.inference_mode or self.frozen:
            # Do not update the statistics during inference or if frozen
            return

        if state is not None:
            self.__update_rms_impl(state, self.state_rms, self._mirror_state, indices)
        if self._observation_is_subset_of_state is not None:
            self.observation_rms.mean.copy_(self.state_rms.mean[self._observation_is_subset_of_state])
            self.observation_rms.var.copy_(self.state_rms.var[self._observation_is_subset_of_state])
            self.observation_rms.std.copy_(self.state_rms.std[self._observation_is_subset_of_state])
            self.observation_rms.count = self.state_rms.count
        else:
            self.__update_rms_impl(observation, self.observation_rms, self._mirror_observation, indices)

    def __update_rms_impl(
        self,
        observation: torch.Tensor,
        rms: RunningMeanStd,
        mirror: SymmetryDef | None = None,
        indices: torch.Tensor | None = None,
    ):
        if indices is not None:
            observation = observation[indices]
        mean, var, count = mean_var_count(observation)
        if mirror is not None:
            mirrored_mean = mirror(mean)
            mirrored_var = abs(mirror(var))
            var = (var + mirrored_var) / 2 + (mean - mirrored_mean) ** 2 / 4
            mean = (mean + mirrored_mean) / 2
        rms.update_from_stats(mean, var, count, synchronize=not self.defer_synchronization)

    def pre_update(self, buffer):
        if self.defer_synchronization:
            if self.state_rms is not None:
                self.state_rms.synchronize()
            if self._observation_is_subset_of_state is not None:
                self.observation_rms.mean.copy_(self.state_rms.mean[self._observation_is_subset_of_state])
                self.observation_rms.var.copy_(self.state_rms.var[self._observation_is_subset_of_state])
                self.observation_rms.std.copy_(self.state_rms.std[self._observation_is_subset_of_state])
                self.observation_rms.count = self.state_rms.count
            else:
                self.observation_rms.synchronize()

    def pre_export(self, graph: ExportGraph):
        graph.add_module_to_graph(
            self.observation_rms,
            module_name="observation_rms",
            input_names={"input": "observation"},
            output_names="observation",
            expose_outputs=False,
        )
