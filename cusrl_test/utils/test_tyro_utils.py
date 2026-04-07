from dataclasses import dataclass

import pytest
import torch
from torch import nn

import cusrl
from cusrl.utils.dataclass_utils import to_strict_typed_dataclass
from cusrl.utils.tyro_utils import cli


def test_tyro_cli_parses_autocast_dtype():
    config = cusrl.preset.ppo.AgentFactory().to_underlying()
    parsed = cli(type(config), default=config, args=["--autocast", "bfloat16"])
    assert parsed.autocast is torch.bfloat16


def test_tyro_cli_parses_prefixed_autocast_dtype():
    config = cusrl.preset.ppo.AgentFactory().to_underlying()
    parsed = cli(type(config), default=config, args=["--autocast", "torch.float16"])
    assert parsed.autocast is torch.float16


def test_tyro_cli_preserves_autocast_bool():
    config = cusrl.preset.ppo.AgentFactory().to_underlying()
    parsed = cli(type(config), default=config, args=["--autocast", "True"])
    assert parsed.autocast is True


def test_tyro_cli_parses_nested_dataclass_factory_field():
    config = cusrl.preset.ppo.AgentFactory().to_underlying()
    parsed = cli(type(config), default=config, args=["--actor-factory.backbone-factory.dropout", "0.25"])
    assert isinstance(parsed.actor_factory, cusrl.nn.ActorFactory)
    assert parsed.actor_factory.backbone_factory.dropout == pytest.approx(0.25)


def test_tyro_cli_parses_named_hook_fields():
    config = cusrl.preset.ppo.AgentFactory().to_underlying()
    parsed = cli(
        type(config),
        default=config,
        args=[
            "--hooks.entropy-loss.weight",
            "0.02",
            "--hooks.generalized-advantage-estimation.gamma",
            "0.95",
        ],
    )
    assert isinstance(parsed.hooks, cusrl.template.actor_critic.HookList)
    assert parsed.hooks.entropy_loss.weight == pytest.approx(0.02)
    assert parsed.hooks.generalized_advantage_estimation.gamma == pytest.approx(0.95)


def test_tyro_cli_help_lists_named_hook_fields(capsys):
    config = cusrl.preset.ppo.AgentFactory().to_underlying()
    with pytest.raises(SystemExit):
        cli(type(config), default=config, args=["--help"])
    captured = capsys.readouterr()
    help_text = captured.out + captured.err
    assert "--hooks.entropy-loss.weight" in help_text
    assert "--hooks.generalized-advantage-estimation.gamma" in help_text
    assert "The coefficient for the entropy loss term." in help_text
    assert "Discount factor for future rewards" in help_text


def test_tyro_cli_parses_torch_module_type():
    @dataclass
    class _Config:
        module_type: type[nn.Module] = nn.Linear

    parsed = cli(_Config, default=_Config(), args=["--module-type", "Linear"])
    assert parsed.module_type is nn.Linear

    parsed = cli(_Config, default=_Config(), args=["--module-type", "torch.nn.ReLU"])
    assert parsed.module_type is nn.ReLU


def test_tyro_cli_rejects_invalid_torch_module_type():
    @dataclass
    class _Config:
        module_type: type[nn.Module] = nn.Linear

    with pytest.raises(SystemExit):
        cli(_Config, default=_Config(), args=["--module-type", "NotAModule"])


def test_strict_typed_dataclass_preserves_original_methods():
    config = cusrl.preset.ppo.AgentFactory().to_underlying()
    strict = to_strict_typed_dataclass(config)
    assert isinstance(strict, type(config))
    assert isinstance(strict.actor_factory, type(config.actor_factory))
    assert callable(strict.register_hook)
    assert callable(strict.actor_factory.__call__)


def test_agent_accepts_prefixed_autocast_dtype():
    class _DummyAgent(cusrl.template.agent.Agent):
        def act(self, observation, state=None):
            raise NotImplementedError

        def step(self, next_observation, reward, terminated, truncated, next_state=None, **kwargs):
            raise NotImplementedError

        def update(self):
            raise NotImplementedError

    environment_spec = cusrl.template.environment.EnvironmentSpec(
        observation_dim=1,
        action_dim=1,
        state_dim=None,
        reward_dim=1,
        num_instances=1,
    )
    agent = _DummyAgent(environment_spec=environment_spec, num_steps_per_update=1, autocast="torch.float16")
    assert agent.autocast_enabled is True
    assert agent.dtype is torch.float16
