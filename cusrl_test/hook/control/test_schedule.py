from types import SimpleNamespace

import pytest

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
