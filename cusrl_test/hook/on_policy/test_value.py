import torch

import cusrl


def test_value_computation_preserves_truncated_next_critic_memory():
    env_spec = cusrl.EnvironmentSpec(num_instances=2, observation_dim=4, action_dim=2)
    agent = cusrl.preset.RecurrentPpoAgentFactory(
        num_steps_per_update=2,
        sampler_epochs=1,
        sampler_mini_batches=1,
    )(env_spec)
    value_hook = agent.hook["value_computation"]

    obs = torch.ones(2, 4, device=agent.device)
    agent.act(obs)
    expected_next_memory = cusrl.utils.nest.map_nested(torch.clone, agent.transition["next_critic_memory"])

    terminated = torch.tensor([[False], [False]], device=agent.device)
    truncated = torch.tensor([[True], [False]], device=agent.device)
    agent.step(obs, torch.zeros(2, 1, device=agent.device), terminated, truncated)

    for (_, buf_mem), (_, exp_mem) in zip(
        cusrl.utils.nest.iterate_nested(agent.buffer["next_critic_memory"]),
        cusrl.utils.nest.iterate_nested(expected_next_memory),
        strict=True,
    ):
        assert torch.allclose(buf_mem[0], exp_mem)
        assert not torch.all(buf_mem[0, 0] == 0)

    for _, curr_mem in cusrl.utils.nest.iterate_nested(value_hook._critic_memory):
        assert torch.all(curr_mem[0] == 0)
        assert not torch.all(curr_mem[1] == 0)

    agent.act(obs)
    agent.step(
        obs,
        torch.zeros(2, 1, device=agent.device),
        torch.zeros(2, 1, dtype=torch.bool, device=agent.device),
        torch.zeros(2, 1, dtype=torch.bool, device=agent.device),
    )
    value_hook.pre_update(agent.buffer)

    expected_truncated_value = agent.critic.evaluate(
        obs[:1],
        memory=cusrl.utils.nest.map_nested(lambda mem: mem[:1], expected_next_memory),
    )
    assert torch.allclose(agent.buffer["next_value"][0, :1], expected_truncated_value)
