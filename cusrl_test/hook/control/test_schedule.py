from types import SimpleNamespace

import pytest
import torch

import cusrl


class TargetHook(cusrl.Hook):
    def __init__(self):
        super().__init__()
        self.weight = 1.0
        self.register_mutable("weight")


def test_hook_parameter_schedule_updates_mutable_attribute_and_records_float():
    target = TargetHook().name_("target")
    schedule = cusrl.hook.HookParameterSchedule("target", "weight", lambda iteration: iteration * 0.25)
    records = {}
    schedule.agent = SimpleNamespace(hook={"target": target}, record=lambda **kwargs: records.update(kwargs))

    schedule.init()
    schedule.apply_schedule(8)

    assert target.weight == 2.0
    assert records == {"target_weight": 2.0}


def test_hook_activation_schedule_toggles_target_hook():
    target = TargetHook().name_("target")
    schedule = cusrl.hook.HookActivationSchedule("target", lambda iteration: iteration % 2 == 0)
    schedule.agent = SimpleNamespace(hook={"target": target})

    schedule.init()
    schedule.apply_schedule(1)
    assert target.active is False

    schedule.apply_schedule(2)
    assert target.active is True


@pytest.mark.parametrize(
    "schedule",
    [
        cusrl.hook.HookParameterSchedule("missing", "weight", lambda iteration: 1.0),
        cusrl.hook.HookActivationSchedule("missing", lambda iteration: True),
    ],
)
def test_hook_schedules_raise_clear_error_for_missing_target(schedule):
    schedule.agent = SimpleNamespace(hook={})

    with pytest.raises(ValueError, match="No hook named 'missing'"):
        schedule.init()


def test_actor_critic_update_and_set_iteration_record_current_iteration_schedule_metrics():
    env_spec = cusrl.EnvironmentSpec(observation_dim=2, action_dim=1)
    agent_factory = cusrl.preset.PpoAgentFactory(
        num_steps_per_update=2, sampler_epochs=1, sampler_mini_batches=1
    ).to_underlying()
    agent_factory.register_hook(
        cusrl.hook.HookParameterSchedule("entropy_loss", "weight", lambda iteration: float(iteration + 1)),
        after="entropy_loss",
    )

    agent = agent_factory(env_spec)
    obs = torch.zeros(1, 2, device=agent.device)
    reward = torch.zeros(1, 1, device=agent.device)
    done = torch.zeros(1, 1, dtype=torch.bool, device=agent.device)
    for _ in range(2):
        agent.act(obs)
        agent.step(obs, reward, done, done)
    metrics_0 = agent.update()
    assert metrics_0["Agent/entropy_loss_weight"] == pytest.approx(1.0)
    assert agent.hook["entropy_loss"].weight == pytest.approx(2.0)

    for _ in range(2):
        agent.act(obs)
        agent.step(obs, reward, done, done)
    metrics_1 = agent.update()
    assert metrics_1["Agent/entropy_loss_weight"] == pytest.approx(2.0)

    resumed_agent = agent_factory(env_spec)
    resumed_agent.set_iteration(5)
    for _ in range(2):
        resumed_agent.act(obs)
        resumed_agent.step(obs, reward, done, done)
    metrics_5 = resumed_agent.update()
    assert metrics_5["Agent/entropy_loss_weight"] == pytest.approx(6.0)
