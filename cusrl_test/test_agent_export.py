from datetime import datetime

import pytest
import torch

import cusrl
from cusrl_test import create_dummy_env

dirname = f"/tmp/cusrl/export/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"


def test_export_ppo_agent():
    environment = create_dummy_env()
    agent = cusrl.preset.ppo.AgentFactory().from_environment(environment)
    agent.export(f"{dirname}/test_export_ppo_agent")


def test_export_recurrent_agent():
    environment = create_dummy_env(with_state=True)
    agent = cusrl.preset.ppo.RecurrentAgentFactory().from_environment(environment)
    agent.export(f"{dirname}/test_export_recurrent_agent", export_critic=True)


@pytest.mark.parametrize("rnn_type", ["LSTM", "GRU"])
def test_export_agent_with_hooks(rnn_type):
    environment = create_dummy_env(with_state=True)
    environment.spec.observation_stats = (
        torch.randn(environment.observation_dim),
        torch.randn(environment.observation_dim) ** 2,
    )
    environment.spec.action_stats = (
        torch.randn(environment.action_dim),
        torch.randn(environment.action_dim) ** 2,
    )
    agent_factory = cusrl.preset.ppo.RecurrentAgentFactory(rnn_type=rnn_type)
    agent_factory.register_hook(cusrl.hook.ReturnPrediction(), index=0)
    agent_factory.register_hook(cusrl.hook.StatePrediction(slice(16, 24)))
    agent_factory.register_hook(cusrl.hook.NextStatePrediction(slice(16, 24)))
    agent = agent_factory.from_environment(environment)
    agent.export(f"{dirname}/test_export_{rnn_type}_agent_with_hook", export_critic=True)
