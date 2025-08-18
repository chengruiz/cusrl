from datetime import datetime

import pytest
import torch

import cusrl
from cusrl_test import create_dummy_env

dirname = f"/tmp/cusrl/export/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"


def test_export_ppo_agent():
    environment = create_dummy_env()
    agent = cusrl.preset.ppo.AgentFactory().from_environment(environment)
    agent.export(f"{dirname}/ppo_agent", target_format="onnx")
    agent.export(f"{dirname}/ppo_agent", target_format="jit")


def test_export_recurrent_ppo_agent():
    environment = create_dummy_env(with_state=True)
    agent = cusrl.preset.ppo.RecurrentAgentFactory().from_environment(environment)
    agent.export(f"{dirname}/recurrent_ppo_agent", target_format="onnx")
    agent.export(f"{dirname}/recurrent_ppo_agent", target_format="jit")


@pytest.mark.parametrize("rnn_type", ["LSTM", "GRU"])
def test_export_agent_with_hooks(rnn_type):
    environment = create_dummy_env(with_state=True)
    environment.spec.observation_normalization = (
        torch.randn(environment.observation_dim).abs(),
        torch.randn(environment.observation_dim),
    )
    environment.spec.action_denormalization = (
        torch.randn(environment.action_dim).abs(),
        torch.randn(environment.action_dim),
    )
    agent_factory = cusrl.preset.ppo.RecurrentAgentFactory(rnn_type=rnn_type)
    agent_factory.register_hook(cusrl.hook.ReturnPrediction(), after="entropy_loss")
    agent_factory.register_hook(cusrl.hook.StatePrediction(slice(16, 24)), after="return_prediction")
    agent_factory.register_hook(cusrl.hook.NextStatePrediction(slice(16, 24)), after="state_prediction")
    agent = agent_factory.from_environment(environment)
    agent.export(f"{dirname}/{rnn_type}_agent_with_hook", target_format="onnx")
    agent.export(f"{dirname}/{rnn_type}_agent_with_hook", target_format="jit")
