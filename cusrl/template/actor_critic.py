import itertools
import os
from collections.abc import Iterable
from typing import Any, Literal

import torch
from typing_extensions import Self

from cusrl.module import Actor, Denormalization, GraphBuilder, Normalization, Value
from cusrl.template.agent import Agent, AgentFactory
from cusrl.template.buffer import Buffer, Sampler
from cusrl.template.environment import EnvironmentSpec
from cusrl.template.hook import Hook, HookComposite
from cusrl.template.optimizer import OptimizerFactory
from cusrl.utils.nest import map_nested
from cusrl.utils.typing import ArrayType, Nested, NestedArray, NestedTensor, Observation, State

__all__ = ["ActorCritic"]


class HookList(list[Hook]):
    def to_dict(self):
        return {hook.name: hook for hook in self}

    @classmethod
    def from_dict(cls, data: dict[str, Hook]) -> "HookList":
        return cls(hook.name_(name) for name, hook in data.items())

    def __getattr__(self, name: str) -> Any:
        for hook in self:
            if hook.name == name:
                return hook
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'.")


class ActorCriticFactory(AgentFactory["ActorCritic"]):
    def __init__(
        self,
        actor_factory: Actor.Factory,
        critic_factory: Value.Factory,
        optimizer_factory: OptimizerFactory,
        sampler: Sampler,
        hooks: Iterable[Hook],
        num_steps_per_update: int,
        name: str = "Agent",
        device: torch.device | str | None = None,
        compile: bool = False,
        autocast: bool | str | torch.dtype = False,
    ):
        super().__init__(
            num_steps_per_update=num_steps_per_update,
            name=name,
            device=device,
            compile=compile,
            autocast=autocast,
        )
        self.actor_factory = actor_factory
        self.critic_factory = critic_factory
        self.optimizer_factory = optimizer_factory
        self.sampler = sampler
        self.hooks = HookList(hooks)

    def __call__(self, environment_spec: EnvironmentSpec):
        return ActorCritic(
            environment_spec=environment_spec,
            actor_factory=self.actor_factory,
            critic_factory=self.critic_factory,
            optimizer_factory=self.optimizer_factory,
            sampler=self.sampler,
            hooks=self.hooks,
            num_steps_per_update=self.num_steps_per_update,
            name=self.name,
            device=self.device,
            compile=self.compile,
            autocast=self.autocast,
        )

    def register_hook(
        self,
        hook: Hook,
        index: int | None = None,
        before: str | None = None,
        after: str | None = None,
    ) -> Self:
        if (index is not None) + (before is not None) + (after is not None) > 1:
            raise ValueError("Only one of index, before, or after can be specified.")

        if before is not None:
            index = self.get_hook_index(before)
        elif after is not None:
            index = self.get_hook_index(after) + 1
        elif index is None:
            index = len(self.hooks)
        self.hooks.insert(index, hook)
        return self

    def get_hook(self, hook_name: str) -> Hook:
        """Gets a registered hook by its name."""
        return self.hooks[self.get_hook_index(hook_name)]

    def get_hook_index(self, hook_name: str) -> int:
        """Gets the index of a registered hook by its name."""
        for i, hook in enumerate(self.hooks):
            if hook.name == hook_name:
                return i
        raise ValueError(f"Hook '{hook_name}' not found.")


class ActorCritic(Agent):
    Factory = ActorCriticFactory
    MODULES = ["actor", "critic", "hook"]
    OPTIMIZERS = ["optimizer"]

    def __init__(
        self,
        environment_spec: EnvironmentSpec,
        actor_factory: Actor.Factory,
        critic_factory: Value.Factory,
        optimizer_factory: OptimizerFactory,
        sampler: Sampler,
        hooks: Iterable[Hook],
        num_steps_per_update: int,
        name: str = "Agent",
        device: torch.device | str | None = None,
        compile: bool = False,
        autocast: bool | torch.dtype = False,
    ):
        super().__init__(
            environment_spec=environment_spec,
            num_steps_per_update=num_steps_per_update,
            name=name,
            device=device,
            compile=compile,
            autocast=autocast,
        )

        self.value_dim = self.environment_spec.reward_dim
        self.buffer_capacity = num_steps_per_update
        self.actor_factory = actor_factory
        self.critic_factory = critic_factory
        self.optimizer_factory = optimizer_factory

        self.hook = HookComposite(hooks)
        self.hook.pre_init(self)
        self.actor: Actor = self.actor_factory(self.observation_dim, self.action_dim)
        self.critic: Value = self.critic_factory(
            self.state_dim + self.action_dim * self.critic_factory.action_aware, self.value_dim
        )
        self.buffer = Buffer(self.buffer_capacity, self.parallelism, device=self.device)
        self.sampler = sampler
        self.grad_scaler = torch.GradScaler(enabled=self.autocast_enabled)

        self.actor_memory = None
        self.hook.init()

        self.actor = self.setup_module(self.actor)
        self.critic = self.setup_module(self.critic)
        if self.compile:
            self.actor.compile()
            self.critic.compile()
            self.hook.compile()
            self._train_step = torch.compile(self._train_step)
        self.optimizer = self.optimizer_factory(
            itertools.chain(
                self.actor.named_parameters(prefix="actor"),
                self.critic.named_parameters(prefix="critic"),
                self.hook.named_parameters(prefix="hook"),
            )
        )
        self._set_training_mode(False)
        self.hook.post_init()
        self.hook.apply_schedule(0)

    @torch.no_grad()
    @Agent._decorator_act__preserve_io_format
    def act(self, observation: Observation, state: State = None):
        self.transition.clear()
        self._save_transition(observation=observation, state=state)
        # enable hook to preprocess the observation and state
        self.hook.pre_act(self.transition)

        with self.autocast():
            action_dist, (action, action_logp), next_actor_memory = self.actor.explore(
                self.transition["observation"].unsqueeze(0),
                memory=self.actor_memory,
                deterministic=self.deterministic,
            )

            action_dist = map_nested(lambda x: x.squeeze(0), action_dist)
            action = action.squeeze(0)
            action_logp = action_logp.squeeze(0)

        self._save_transition(
            actor_memory=self.actor_memory,
            action_dist=action_dist,
            action=action,
            action_logp=action_logp,
        )
        self.actor_memory = next_actor_memory

        # enable hook to postprocess the action
        self.hook.post_act(self.transition)
        return self.transition["action"]

    @torch.no_grad()
    def step(
        self,
        next_observation: ArrayType,
        reward: ArrayType,
        terminated: ArrayType,
        truncated: ArrayType,
        next_state: ArrayType | None = None,
        **kwargs: Nested[ArrayType],
    ) -> bool:
        self._save_transition(
            next_observation=next_observation,
            next_state=next_state,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            done=terminated | truncated,
            **kwargs,
        )
        if self.transition["terminated"].dtype != torch.bool:
            raise TypeError("'terminated' should be of boolean type.")
        if self.transition["truncated"].dtype != torch.bool:
            raise TypeError("'truncated' should be of boolean type.")

        # enable hook to preprocess the next_observation, next_state, etc.
        self.hook.post_step(self.transition)
        if not self.inference_mode:
            self.buffer.push(self.transition)
        self.actor.reset_memory(self.actor_memory, self.transition["done"])
        return super().step(
            next_observation=next_observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            next_state=next_state,
            **kwargs,
        )

    @Agent._decorator_update__set_training_mode
    def update(self):
        self.hook.pre_update(self.buffer)
        for batch in self.sampler(self.buffer):
            self._train_step(batch)
        self.hook.post_update()
        self.hook.apply_schedule(self.iteration + 1)
        return super().update()

    def _train_step(self, batch: dict[str, NestedTensor | Any]):
        if (loss := self.hook.objective(batch)) is not None:
            self.optimizer.zero_grad()
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            self.hook.pre_optim(self.optimizer)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.hook.post_optim()

    def set_iteration(self, iteration: int):
        if iteration != self.iteration:
            super().set_iteration(iteration)
            self.hook.apply_schedule(self.iteration)

    def resize_buffer(self, capacity: int):
        if self.buffer_capacity != capacity:
            self.buffer_capacity = capacity
            self.buffer.resize(capacity)

    def export(
        self,
        output_dir: str,
        target_format: Literal["onnx", "jit"] = "onnx",
        optimize: bool = True,
        batch_size: int = 1,
        dynamo: bool = False,
        verbose: bool = True,
        **kwargs,
    ):
        os.makedirs(output_dir, exist_ok=True)

        actor = self.actor_factory(self.observation_dim, self.action_dim).to(device=self.device)
        actor.load_state_dict(self.actor.state_dict())
        graph = GraphBuilder(graph_name="actor")
        inputs = {"observation": torch.zeros(1, batch_size, self.environment_spec.observation_dim, device=self.device)}
        input_names = {"observation": "observation"}
        output_names = ["action"]
        if actor.is_recurrent:
            _, init_memory = actor(**inputs)
            actor.reset_memory(init_memory)
            inputs["memory_in"] = init_memory
            input_names["memory"] = "memory_in"
            output_names.append("memory_out")

        self.hook.pre_export(graph)
        graph.add_node(
            actor,
            module_name="actor",
            input_names=input_names,
            output_names=output_names,
            extra_kwargs={"forward_type": "act_deterministic"},
            info={
                "observation_dim": self.environment_spec.observation_dim,
                "action_dim": self.action_dim,
                "is_recurrent": actor.is_recurrent,
            },
            expose_outputs=True,
        )
        self.hook.post_export(graph)
        if self.environment_spec.observation_normalization is not None:
            graph.add_node(
                Normalization(
                    self.to_tensor(self.environment_spec.observation_normalization[1]),
                    self.to_tensor(self.environment_spec.observation_normalization[0]),
                ),
                module_name="observation_normalization",
                input_names={"input": "observation"},
                output_names="observation",
                expose_outputs=False,
                prepend=True,
            )
        if self.environment_spec.action_denormalization is not None:
            graph.add_node(
                Denormalization(
                    self.to_tensor(self.environment_spec.action_denormalization[1]),
                    self.to_tensor(self.environment_spec.action_denormalization[0]),
                ),
                module_name="action_denormalization",
                input_names={"input": "action"},
                output_names="action",
                expose_outputs=False,
            )

        if target_format == "onnx":
            graph.export_onnx(inputs, output_dir, optimize=optimize, dynamo=dynamo, verbose=verbose)
        elif target_format == "jit":
            graph.export_jit(inputs, output_dir, optimize=optimize)
        else:
            raise ValueError(f"Unsupported export format '{target_format}'.")
        if verbose:
            print(f"Agent exported to \033[4m{output_dir}\033[0m in '{target_format}' format.")

    def _save_transition(self, **kwargs: NestedArray | None):
        for key, value in kwargs.items():
            if value is None:
                continue
            try:
                self.transition[key] = self.to_nested_tensor(value)
            except Exception as error:
                raise ValueError(f"Failed to convert '{key}' to tensor.") from error
