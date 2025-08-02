from functools import partial

import torch

import cusrl
from cusrl_test import create_dummy_env


def test_amp():
    agent_factory = cusrl.preset.amp.AgentFactory(
        amp_dataset_source=partial(torch.randn, 100, 16),
        amp_state_indices=slice(16, 24),
    )
    cusrl.Trainer(
        partial(create_dummy_env, with_state=True),
        agent_factory,
        num_iterations=5,
    ).run_training_loop()
