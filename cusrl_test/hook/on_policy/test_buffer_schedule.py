from types import SimpleNamespace

import cusrl


def test_on_policy_buffer_capacity_schedule_resizes_agent_buffer():
    calls = []
    agent = SimpleNamespace(num_steps_per_update=0, resize_buffer=lambda capacity: calls.append(capacity))
    hook = cusrl.hook.OnPolicyBufferCapacitySchedule(lambda iteration: iteration + 5)
    hook.agent = agent

    hook.apply_schedule(7)

    assert agent.num_steps_per_update == 12
    assert calls == [12]
