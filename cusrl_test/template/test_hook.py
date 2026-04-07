from cusrl.hook import ConditionalObjectiveActivation, EnvironmentSpecOverride, GeneralizedAdvantageEstimation
from cusrl.utils import from_dict, to_dict


def always_active(agent, metadata, batch):
    return True


def test_hook_round_trip_preserves_constructor_fields():
    hook = GeneralizedAdvantageEstimation(gamma=0.9, lamda=0.8, recompute=True)

    restored = from_dict(None, to_dict(hook))

    assert isinstance(restored, GeneralizedAdvantageEstimation)
    assert restored.gamma == 0.9
    assert restored.lamda == 0.8
    assert restored.recompute is True


def test_hook_round_trip_supports_dict_payload_constructor():
    hook = EnvironmentSpecOverride(overrides={"num_instances": 1024})

    restored = from_dict(None, to_dict(hook))

    assert isinstance(restored, EnvironmentSpecOverride)
    assert restored.overrides == {"num_instances": 1024}


def test_hook_round_trip_supports_named_conditions_constructor():
    hook = ConditionalObjectiveActivation(named_conditions={"value_loss": always_active})

    restored = from_dict(None, to_dict(hook))

    assert isinstance(restored, ConditionalObjectiveActivation)
    assert restored.named_conditions == {"value_loss": always_active}
