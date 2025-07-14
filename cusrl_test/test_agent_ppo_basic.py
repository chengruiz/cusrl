import numpy as np
import objprint
import torch

import cusrl
from cusrl.utils import distributed
from cusrl_test import create_dummy_env, run_environment_evaluation_loop


def test_environment_with_observation_only():
    environment = create_dummy_env()
    agent = cusrl.preset.ppo.AgentFactory().from_environment(environment)
    run_environment_evaluation_loop(environment, agent)


def test_environment_with_observation_and_state():
    environment = create_dummy_env(with_state=True)
    agent = cusrl.preset.ppo.AgentFactory().from_environment(environment)

    objprint.op(agent)
    state_dict = agent.state_dict()

    run_environment_evaluation_loop(environment, agent)

    if distributed.enabled():
        torch.set_printoptions(precision=3)
        distributed.print_all(agent.actor.state_dict()["backbone.layers.0.bias"][:5])

    agent.load_state_dict(state_dict)


def test_environment_with_multiple_rewards():
    environment = create_dummy_env(reward_dim=2)
    agent_factory = cusrl.preset.ppo.AgentFactory()
    agent_factory.register_hook(cusrl.hook.AdvantageReduction(), before="AdvantageNormalization")
    cusrl.Trainer(environment=environment, agent_factory=agent_factory, num_iterations=5).run_training_loop()


def test_numpy_environment():
    environment = create_dummy_env(numpy=True)
    agent = cusrl.preset.ppo.AgentFactory().from_environment(environment)
    run_environment_evaluation_loop(environment, agent)


def test_action_type():
    agent = cusrl.preset.ppo.AgentFactory()(observation_dim=35, action_dim=12)
    action_numpy = agent.act(np.random.randn(8, 35).astype(np.float32))
    action_tensor_cpu = agent.act(torch.randn(8, 35))
    action_tensor_cuda = agent.act(torch.randn(8, 35).to(cusrl.device()))

    assert isinstance(action_numpy, np.ndarray)
    assert isinstance(action_tensor_cpu, torch.Tensor)
    assert action_tensor_cpu.device == torch.device("cpu")
    assert isinstance(action_tensor_cuda, torch.Tensor)
    assert action_tensor_cuda.device == cusrl.device()
