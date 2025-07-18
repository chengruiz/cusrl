import torch

import cusrl
from cusrl_test import create_dummy_env

agent_factory = cusrl.preset.ppo.AgentFactory()

agent_factory.register_hook(
    cusrl.hook.RandomNetworkDistillation(
        module_factory=cusrl.Mlp.Factory(hidden_dims=[128, 64]),
        output_dim=16,
        reward_scale=0.1,
    ),
    before="ValueComputation",
)
agent_factory.register_hook(cusrl.hook.StatePrediction(slice(16, 24)))


agent = agent_factory(35, 12, 42)


def print_state_dict(state_dict):
    for key, val in state_dict.items():
        if isinstance(val, dict):
            print(f"{key}:")
            print_state_dict(val)
        elif isinstance(val, torch.Tensor):
            print(key, val.shape)
        else:
            print(key, type(val).__name__)


def test_load_state_dict():
    state_dict = agent.state_dict()
    print("agent state dict")
    print_state_dict(state_dict)
    agent.load_state_dict(state_dict)


def test_load_empty_state_dict():
    agent.load_state_dict({})


def test_load_state_dict_with_extra_keys():
    state_dict = agent.state_dict()
    state_dict["__unused1"] = None
    state_dict["__unused2"] = None
    agent.load_state_dict(state_dict)


def test_load_state_dict_with_missing_keys():
    state_dict = agent.state_dict()
    state_dict.pop("actor")
    agent.load_state_dict(state_dict)


def test_load_state_dict_with_mismatched_parameter():
    state_dict = agent.state_dict()
    state_dict["actor"]["backbone.layers.0.weight"] = torch.zeros([1])
    agent.load_state_dict(state_dict)


def test_load_state_dict_with_missing_hook():
    state_dict = agent.state_dict()
    state_dict["hook"].pop("RandomNetworkDistillation")
    agent.load_state_dict(state_dict)
    state_dict["hook"].pop("StatePrediction")
    agent.load_state_dict(state_dict)


def test_load_state_dict_with_missing_hook_keys():
    state_dict = agent.state_dict()
    state_dict["hook"]["RandomNetworkDistillation"].pop("predictor")
    agent.load_state_dict(state_dict)


def test_load_state_dict_with_mismatched_hook_parameter():
    state_dict = agent.state_dict()
    state_dict["hook"]["RandomNetworkDistillation"]["predictor"]["layers.0.weight"] = torch.zeros([1])
    agent.load_state_dict(state_dict)


def test_save_state_dict():
    logger = cusrl.Logger("/tmp/cusrl/agent")
    logger.save_checkpoint(agent.state_dict(), 0)
    trial = cusrl.Trial(logger.log_dir)
    agent.load_state_dict(trial.load_checkpoint(agent.device))


def test_trainer_save_state_dict():
    trainer = cusrl.Trainer(
        create_dummy_env(with_state=True),
        agent_factory,
        logger_factory=cusrl.Logger.Factory("/tmp/cusrl/trainer"),
        num_iterations=10,
        save_interval=5,
    )
    trainer.run_training_loop()
    trial = cusrl.Trial(trainer.logger.log_dir)
    assert trial.all_iterations == [0, 5, 10]
