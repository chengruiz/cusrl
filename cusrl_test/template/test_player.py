from collections.abc import Callable

import pytest
import torch

import cusrl


def _clone_value(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_value(val) for key, val in value.items()}
    return value


class RecordingAgent(cusrl.Agent):
    def __init__(self, environment_spec: cusrl.EnvironmentSpec):
        super().__init__(environment_spec=environment_spec, num_steps_per_update=1, device="cpu")
        self.act_inputs: list[tuple[torch.Tensor, torch.Tensor | None]] = []
        self.step_inputs: list[dict[str, torch.Tensor | None]] = []
        self.loaded_state_dict: dict | None = None
        self.inference_mode_calls: list[tuple[bool, bool | None]] = []

    def act(self, observation, state=None):
        observation_tensor = self.to_tensor(observation)
        state_tensor = None if state is None else self.to_tensor(state)
        self.act_inputs.append((observation_tensor.clone(), None if state_tensor is None else state_tensor.clone()))

        action = torch.full(
            (self.parallelism, self.action_dim),
            float(len(self.act_inputs)),
            device=self.device,
        )
        self.transition = {
            "act_call": torch.tensor([len(self.act_inputs)], dtype=torch.long),
            "action": action.clone(),
        }
        return action

    def step(self, next_observation, reward, terminated, truncated, next_state=None, **kwargs):
        step_index = len(self.step_inputs) + 1
        marker = kwargs.get("marker")
        self.step_inputs.append({
            "next_observation": self.to_tensor(next_observation).clone(),
            "next_state": None if next_state is None else self.to_tensor(next_state).clone(),
            "reward": self.to_tensor(reward).clone(),
            "terminated": self.to_tensor(terminated).clone(),
            "truncated": self.to_tensor(truncated).clone(),
            "marker": None if marker is None else self.to_tensor(marker).clone(),
        })
        self.transition = {
            "step_call": torch.tensor([step_index], dtype=torch.long),
            "marker": None if marker is None else self.to_tensor(marker).clone(),
        }
        return super().step(next_observation, reward, terminated, truncated, next_state, **kwargs)

    def update(self):
        return super().update()

    def load_state_dict(self, state_dict: dict):
        self.loaded_state_dict = state_dict

    def set_inference_mode(self, mode: bool = True, deterministic: bool | None = True):
        self.inference_mode_calls.append((mode, deterministic))
        super().set_inference_mode(mode=mode, deterministic=deterministic)


class RecordingAgentFactory:
    def __init__(self):
        self.created_agent: RecordingAgent | None = None

    def from_environment(self, environment: cusrl.Environment) -> RecordingAgent:
        self.created_agent = RecordingAgent(environment.spec)
        return self.created_agent


class ScriptedEnvironment(cusrl.Environment):
    def __init__(
        self,
        *,
        reset_outputs: list[dict],
        step_outputs: list[dict],
        num_instances: int,
        observation_dim: int = 1,
        action_dim: int = 1,
        state_dim: int | None = 1,
        reward_dim: int = 1,
        autoreset: bool = False,
        timestep: float | None = None,
        metrics: dict[str, float] | None = None,
    ):
        super().__init__(
            observation_dim=observation_dim,
            action_dim=action_dim,
            state_dim=state_dim,
            reward_dim=reward_dim,
            num_instances=num_instances,
            autoreset=autoreset,
            timestep=timestep,
        )
        self._reset_outputs = list(reset_outputs)
        self._step_outputs = list(step_outputs)
        self._metrics = {} if metrics is None else metrics
        self.reset_calls: list[list[int] | None] = []
        self.step_actions: list[torch.Tensor] = []
        self.loaded_state_dict: dict | None = None
        self.closed = False

    def reset(self, *, indices=None, randomize_episode_progress=False):
        self.reset_calls.append(None if indices is None else list(indices))
        assert self._reset_outputs, "Unexpected reset call"
        output = self._reset_outputs.pop(0)
        expected_indices = output.get("indices")
        assert expected_indices == (None if indices is None else list(indices))
        return (
            _clone_value(output["observation"]),
            _clone_value(output["state"]),
            _clone_value(output.get("info", {})),
        )

    def step(self, action):
        self.step_actions.append(action.clone())
        assert self._step_outputs, "Unexpected step call"
        output = self._step_outputs.pop(0)
        return (
            _clone_value(output["observation"]),
            _clone_value(output["state"]),
            _clone_value(output["reward"]),
            _clone_value(output["terminated"]),
            _clone_value(output["truncated"]),
            _clone_value(output.get("info", {})),
        )

    def get_metrics(self) -> dict[str, float]:
        return dict(self._metrics)

    def load_state_dict(self, state_dict: dict):
        self.loaded_state_dict = state_dict

    def close(self):
        self.closed = True


class RecordingHook(cusrl.template.PlayerHook):
    def __init__(self):
        self.initialized = False
        self.step_events: list[tuple[int, int]] = []
        self.reset_events: list[list[int]] = []
        self.closed = False

    def init(self, player: cusrl.Player):
        super().init(player)
        self.initialized = True

    def step(self, step: int, transition: dict[str, torch.Tensor]):
        self.step_events.append((step, int(transition["step_call"].item())))

    def reset(self, indices):
        self.reset_events.append(list(indices))

    def close(self):
        self.closed = True


class EnvironmentFactory:
    def __init__(self, builder: Callable[[], ScriptedEnvironment]):
        self.builder = builder
        self.created_environment: ScriptedEnvironment | None = None

    def __call__(self) -> ScriptedEnvironment:
        self.created_environment = self.builder()
        return self.created_environment


def test_player_loads_checkpoint_and_builds_from_factories(tmp_path):
    def build_environment():
        return ScriptedEnvironment(
            num_instances=2,
            timestep=0.25,
            reset_outputs=[{
                "indices": None,
                "observation": torch.tensor([[1.0], [2.0]]),
                "state": torch.tensor([[3.0], [4.0]]),
            }],
            step_outputs=[],
        )

    environment_factory = EnvironmentFactory(build_environment)
    agent_factory = RecordingAgentFactory()

    trial_dir = tmp_path / "DummyEnv_ppo" / "trial_0"
    checkpoint_dir = trial_dir / "ckpt"
    checkpoint_dir.mkdir(parents=True)
    torch.save({"agent": {"restored": 7}, "environment": {"seed": 11}}, checkpoint_dir / "ckpt_3.pt")

    player = cusrl.Player(
        environment=environment_factory,
        agent=agent_factory,
        checkpoint_path=str(trial_dir),
        num_steps=0,
        deterministic=False,
        verbose=False,
    )

    assert environment_factory.created_environment is not None
    assert agent_factory.created_agent is not None
    assert player.environment is environment_factory.created_environment
    assert player.agent is agent_factory.created_agent
    assert player.trial is not None
    assert player.timestep == 0.25
    assert player.agent.loaded_state_dict == {"restored": 7}
    assert player.environment.loaded_state_dict == {"seed": 11}
    assert player.agent.inference_mode is True
    assert player.agent.deterministic is False
    assert player.agent.inference_mode_calls == [(True, False)]


def test_player_runs_num_steps_invokes_hooks_and_reports_metrics():
    environment = ScriptedEnvironment(
        num_instances=2,
        timestep=None,
        metrics={"environment/score": 3.0},
        reset_outputs=[{
            "indices": None,
            "observation": torch.tensor([[0.0], [1.0]]),
            "state": torch.tensor([[10.0], [11.0]]),
        }],
        step_outputs=[
            {
                "observation": torch.tensor([[2.0], [3.0]]),
                "state": torch.tensor([[12.0], [13.0]]),
                "reward": torch.tensor([[1.0], [3.0]]),
                "terminated": torch.tensor([[False], [False]]),
                "truncated": torch.tensor([[False], [False]]),
                "info": {"marker": torch.tensor([[101.0], [102.0]])},
            },
            {
                "observation": torch.tensor([[4.0], [5.0]]),
                "state": torch.tensor([[14.0], [15.0]]),
                "reward": torch.tensor([[5.0], [7.0]]),
                "terminated": torch.tensor([[False], [False]]),
                "truncated": torch.tensor([[False], [False]]),
                "info": {"marker": torch.tensor([[201.0], [202.0]])},
            },
        ],
    )
    agent = RecordingAgent(environment.spec)
    hook = RecordingHook()

    metrics = cusrl.Player(environment, agent, num_steps=2, verbose=False, hooks=[hook]).run_playing_loop()

    assert hook.initialized is True
    assert hook.step_events == [(0, 1), (1, 2)]
    assert hook.reset_events == []
    assert hook.closed is True
    assert environment.closed is True
    assert len(agent.act_inputs) == 2
    assert len(environment.step_actions) == 2
    assert metrics["Mean step reward"] == pytest.approx(4.0)
    assert metrics["Mean episode reward"] == 0.0
    assert metrics["Mean episode length"] == 0.0
    assert metrics["environment/score"] == 3.0


def test_player_resets_done_instances_and_updates_next_agent_inputs():
    environment = ScriptedEnvironment(
        num_instances=2,
        timestep=None,
        reset_outputs=[
            {
                "indices": None,
                "observation": torch.tensor([[100.0], [200.0]]),
                "state": torch.tensor([[300.0], [400.0]]),
            },
            {
                "indices": [0],
                "observation": torch.tensor([[900.0]]),
                "state": torch.tensor([[901.0]]),
            },
        ],
        step_outputs=[
            {
                "observation": torch.tensor([[1.0], [2.0]]),
                "state": torch.tensor([[10.0], [20.0]]),
                "reward": torch.tensor([[0.0], [0.0]]),
                "terminated": torch.tensor([[True], [False]]),
                "truncated": torch.tensor([[False], [False]]),
                "info": {"marker": torch.tensor([[1.0], [2.0]])},
            },
            {
                "observation": torch.tensor([[3.0], [4.0]]),
                "state": torch.tensor([[30.0], [40.0]]),
                "reward": torch.tensor([[0.0], [0.0]]),
                "terminated": torch.tensor([[False], [False]]),
                "truncated": torch.tensor([[False], [False]]),
                "info": {"marker": torch.tensor([[3.0], [4.0]])},
            },
        ],
    )
    agent = RecordingAgent(environment.spec)
    hook = RecordingHook()
    player = cusrl.Player(environment, agent, num_steps=2, verbose=False, hooks=[hook])

    player.run_playing_loop()

    second_observation, second_state = agent.act_inputs[1]
    assert environment.reset_calls == [None, [0]]
    assert torch.equal(second_observation, torch.tensor([[900.0], [2.0]]))
    assert second_state is not None
    assert torch.equal(second_state, torch.tensor([[901.0], [20.0]]))
    assert hook.reset_events == [[0]]
    assert player.episode_count.tolist() == [1, 0]


def test_player_stops_after_each_instance_reaches_num_episodes():
    environment = ScriptedEnvironment(
        num_instances=2,
        timestep=None,
        reset_outputs=[
            {
                "indices": None,
                "observation": torch.tensor([[0.0], [10.0]]),
                "state": torch.tensor([[1.0], [11.0]]),
            },
            {
                "indices": [0],
                "observation": torch.tensor([[20.0]]),
                "state": torch.tensor([[21.0]]),
            },
            {
                "indices": [1],
                "observation": torch.tensor([[30.0]]),
                "state": torch.tensor([[31.0]]),
            },
        ],
        step_outputs=[
            {
                "observation": torch.tensor([[2.0], [12.0]]),
                "state": torch.tensor([[3.0], [13.0]]),
                "reward": torch.tensor([[1.0], [1.0]]),
                "terminated": torch.tensor([[True], [False]]),
                "truncated": torch.tensor([[False], [False]]),
                "info": {},
            },
            {
                "observation": torch.tensor([[4.0], [14.0]]),
                "state": torch.tensor([[5.0], [15.0]]),
                "reward": torch.tensor([[1.0], [1.0]]),
                "terminated": torch.tensor([[False], [True]]),
                "truncated": torch.tensor([[False], [False]]),
                "info": {},
            },
        ],
    )
    agent = RecordingAgent(environment.spec)
    player = cusrl.Player(environment, agent, num_steps=10, num_episodes=1, verbose=False)

    player.run_playing_loop()

    assert player.step_count == 2
    assert player.episode_count.tolist() == [1, 1]
    assert environment.reset_calls == [None, [0], [1]]
