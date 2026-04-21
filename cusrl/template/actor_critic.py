import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

import torch
from typing_extensions import Self

from cusrl.nn.layer import FlowGraph
from cusrl.nn.module import Actor, ActorFactory, Denormalization, Normalization, Value, ValueFactory
from cusrl.template.agent import Agent, AgentFactory
from cusrl.template.buffer import Buffer, Sampler
from cusrl.template.environment import EnvironmentSpec
from cusrl.template.hook import Hook, HookComposite
from cusrl.template.optimizer import OptimizerFactory
from cusrl.utils.dict_utils import from_dict, to_dict
from cusrl.utils.distributed import broadcast_parameters, reduce_gradients
from cusrl.utils.typing import ArrayT, Nested, NestedArray, NestedTensor, Observation, State

__all__ = ["ActorCritic", "ActorCriticFactory", "HookList"]


class HookList(list[Hook]):
    """A specialized list for managing a collection of Hook objects.

    This class extends the built-in `list` to provide additional functionality
    for handling `Hook` instances. It allows for converting the list to and from
    a dictionary representation and enables accessing individual hooks by their
    name as if they were attributes of the list."""

    def to_dict(self):
        """Converts the list of hooks into a dictionary, keyed by hook names."""
        return {hook.name: hook for hook in self}

    @classmethod
    def from_dict(cls, data: dict[str, Hook]) -> "HookList":
        """Creates a HookList instance from a dictionary."""
        return cls(hook.name_(name) for name, hook in data.items())

    def __getattr__(self, name: str) -> Any:
        """Enables attribute-style access to hooks by name."""
        for hook in self:
            if hook.name == name:
                return hook
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @classmethod
    def coerce(cls, data: Any) -> "HookList":
        """Normalizes serialized / CLI-expanded hook data back into HookList."""
        if isinstance(data, cls):
            return cls(data)
        if isinstance(data, (list, tuple)):
            return cls(data)

        restored = from_dict(None, to_dict(data))
        if isinstance(restored, cls):
            return restored
        if isinstance(restored, dict):
            return cls.from_dict(restored)
        if isinstance(restored, (list, tuple)):
            return cls(restored)
        raise TypeError(f"Unsupported hooks payload: {type(data)!r}")


@dataclass(kw_only=True)
class ActorCriticFactory(AgentFactory["ActorCritic"]):
    actor_factory: ActorFactory
    """The factory for creating the actor module."""
    critic_factory: ValueFactory
    """The factory for creating the critic module."""
    optimizer_factory: OptimizerFactory
    """The factory for creating the optimizer."""
    sampler: Sampler
    """The sampler for generating training batches."""
    hooks: list[Hook]
    """A list of hooks to be called during the agent's lifecycle."""

    def __post_init__(self):
        self.hooks = HookList.coerce(self.hooks)

    def __call__(self, environment_spec: EnvironmentSpec):
        """Instantiate an Actor-Critic agent with the given environment spec."""
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
        """Registers a hook to be called during the agent's lifecycle.

        The hook can be inserted at a specific position in the execution order.
        The position can be specified by an index, or relative to another hook
        by its name. If no position is specified, the hook is appended to the
        end of the list.

        Note:
            Only one of ``index``, ``before``, or ``after`` can be specified
            at a time.

        Args:
            hook (Hook):
                The hook instance to register.
            index (int | None, optional):
                The index at which to insert the hook. Defaults to ``None``.
            before (str | None, optional):
                The name of an existing hook to insert this hook before.
                Defaults to ``None``.
            after (str | None, optional):
                The name of an existing hook to insert this hook after. Defaults
                to ``None``.
        """
        if (index is not None) + (before is not None) + (after is not None) > 1:
            raise ValueError("Only one of index, before, or after can be specified")

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
        raise ValueError(f"No hook named '{hook_name}' is registered")


class ActorCritic(Agent):
    """An agent that implements the Actor-Critic algorithm.

    This class combines an actor and a critic to learn a policy and a value
    function. The actor determines the agent's actions, while the critic
    evaluates them. The agent interacts with the environment, stores its
    experiences in a buffer, and periodically updates its networks based on
    sampled data.

    The agent's behavior can be customized and extended through a system of
    hooks, which can be inserted at various points in the agent's lifecycle
    (e.g., before and after initialization, acting, or updating).

    The agent supports both recurrent and non-recurrent models, and can be
    configured for mixed-precision training and just-in-time (JIT) compilation
    for performance optimization.
    """

    Factory = ActorCriticFactory
    MODULES = ["actor", "critic", "hook"]
    STATEFULS = ["optimizer", "grad_scaler"]

    def __init__(
        self,
        environment_spec: EnvironmentSpec,
        actor_factory: ActorFactory,
        critic_factory: ValueFactory,
        optimizer_factory: OptimizerFactory,
        sampler: Sampler,
        hooks: Iterable[Hook],
        num_steps_per_update: int,
        name: str = "Agent",
        device: torch.device | str | None = None,
        compile: bool | str = False,
        autocast: bool | None | torch.dtype | str = False,
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
        self.grad_scaler = torch.GradScaler(device=str(self.device), enabled=self.autocast_enabled)

        self.actor_memory = None
        self.hook.init()

        self.actor = self.setup_module(self.actor)
        self.critic = self.setup_module(self.critic)
        if self.compile:
            self.actor.compile(**self._get_compile_kwargs())
            self.critic.compile(**self._get_compile_kwargs())
            self.hook.compile(**self._get_compile_kwargs())
        self.optimizer = self.optimizer_factory(self.named_parameters())
        self._set_training_mode(False)
        self.hook.post_init()
        broadcast_parameters(self.parameters())
        self.hook.apply_schedule(0)

    @torch.no_grad()
    @Agent._decorator_act__preserve_io_format
    def act(self, observation: Observation, state: State = None):
        self.transition.clear()
        self._save_transition(observation=observation, state=state)
        # enable hook to preprocess the observation and state
        self.hook.pre_act(self.transition)

        action_dist, (action, action_logp), next_actor_memory = self.actor.explore(
            self.transition["observation"],
            memory=self.actor_memory,
            deterministic=self.deterministic,
            backbone_kwargs={"sequential": False},
        )

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
        next_observation: ArrayT,
        reward: ArrayT,
        terminated: ArrayT,
        truncated: ArrayT,
        next_state: ArrayT | None = None,
        **kwargs: Nested[ArrayT],
    ) -> bool:
        self._save_transition(
            next_observation=next_observation,
            next_state=next_state,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            **kwargs,
        )
        if self.transition["terminated"].dtype != torch.bool:
            raise TypeError("'terminated' must have dtype bool")
        if self.transition["truncated"].dtype != torch.bool:
            raise TypeError("'truncated' must have dtype bool")
        self.transition["done"] = self.transition["terminated"] | self.transition["truncated"]

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
        ) and self.hook.should_update(self.transition)

    def update(self):
        self.hook.pre_update(self.buffer)
        with self._training_mode():
            for metadata, batch in self.sampler(self.buffer):
                self._train_step(metadata, batch)
        self.hook.post_update()
        self.hook.apply_schedule(self.iteration + 1)
        return super().update()

    def _train_step(self, metadata: dict[str, Any], batch: dict[str, NestedTensor]):
        self.actor.clear_intermediate_repr()
        self.critic.clear_intermediate_repr()
        self.hook.pre_objective(metadata, batch)
        with self.autocast():
            objectives = self.hook.objective(metadata, batch)
        if objectives is not None:
            loss = sum(objectives.values())
            self.optimizer.zero_grad()
            scaled_loss = self.grad_scaler.scale(loss)
            scaled_loss.backward()
            self.grad_scaler.unscale_(self.optimizer)
            reduce_gradients(self.optimizer)
            self.hook.pre_optim(self.optimizer)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.hook.post_optim()
            self.record(**objectives)
        self.hook.post_objective(metadata, batch)

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
        *,
        target_format: Literal["onnx", "jit"] = "onnx",
        with_environment_normalization: bool = True,
        optimize: bool = True,
        sequence_len: int = 1,
        batch_size: int = 1,
        opset_version: int | None = None,
        dynamo: bool = False,
        verbose: bool = True,
        **kwargs,
    ):
        os.makedirs(output_dir, exist_ok=True)

        graph = FlowGraph(graph_name="actor")
        if with_environment_normalization and self.environment_spec.observation_normalization is not None:
            graph.add_node(
                Normalization(
                    self.to_tensor(self.environment_spec.observation_normalization[1]),
                    self.to_tensor(self.environment_spec.observation_normalization[0]),
                ),
                module_name="observation_normalization",
                input_names={"input": "observation"},
                output_names="observation",
                expose_outputs=False,
            )

        inputs = {}
        inputs["observation"] = torch.zeros(
            sequence_len, batch_size, self.environment_spec.observation_dim, device=self.device
        )
        self.hook.pre_export(graph)

        actor_input_names = {"observation": "observation"}
        actor_output_names = ["action"]
        if self.actor.is_recurrent:
            _, actor_init_memory = self.actor(**inputs)
            self.actor.reset_memory(actor_init_memory)
            inputs["memory_in"] = actor_init_memory
            actor_input_names["memory"] = "memory_in"
            actor_output_names.append("memory_out")

        graph.add_node(
            self.actor,
            module_name="actor",
            input_names=actor_input_names,
            output_names=actor_output_names,
            extra_kwargs={"forward_type": "act_deterministic"},
            info={
                "observation_dim": self.environment_spec.observation_dim,
                "action_dim": self.action_dim,
                "is_recurrent": self.actor.is_recurrent,
            },
            expose_outputs=True,
        )

        self.hook.post_export(graph)

        if with_environment_normalization and self.environment_spec.action_denormalization is not None:
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
            graph.export_onnx(
                inputs,
                output_dir,
                optimize=optimize,
                dynamo=dynamo,
                verbose=verbose,
                opset_version=opset_version,
            )
        elif target_format == "jit":
            graph.export_jit(inputs, output_dir, optimize=optimize)
        else:
            raise ValueError(f"Unsupported export format '{target_format}'")
        if verbose:
            print(f"Agent exported to \033[4m{output_dir}\033[0m in '{target_format}' format.")

    def _save_transition(self, **kwargs: NestedArray | None):
        for key, value in kwargs.items():
            if value is None:
                continue
            try:
                self.transition[key] = self.to_nested_tensor(value)
            except Exception as error:
                raise ValueError(f"Failed to convert transition field '{key}' to a tensor") from error
