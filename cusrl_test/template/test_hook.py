import pytest

from cusrl.hook.condition import ConditionalObjectiveActivation
from cusrl.hook.environment_spec import EnvironmentSpecOverride
from cusrl.hook.gae import GeneralizedAdvantageEstimation
from cusrl.template import HookFactory


def test_hook_factory_get_hook_type_returns_declared_hook_type():
    assert GeneralizedAdvantageEstimation.Factory.get_hook_type() is GeneralizedAdvantageEstimation


def test_hook_factory_get_hook_type_inherits_declared_hook_type():
    class DerivedFactory(GeneralizedAdvantageEstimation.Factory):
        pass

    assert DerivedFactory.get_hook_type() is GeneralizedAdvantageEstimation


def test_hook_factory_call_instantiates_declared_hook_type():
    hook = GeneralizedAdvantageEstimation.Factory(gamma=0.9, lamda=0.8, recompute=True)()

    assert isinstance(hook, GeneralizedAdvantageEstimation)
    assert hook.gamma == 0.9
    assert hook.lamda == 0.8
    assert hook.recompute is True


def test_hook_factory_call_supports_dict_payload_constructor():
    hook = EnvironmentSpecOverride.Factory(overrides={"num_instances": 1024})()

    assert isinstance(hook, EnvironmentSpecOverride)
    assert hook.overrides == {"num_instances": 1024}


def test_hook_factory_call_supports_named_conditions_constructor():
    condition = lambda agent, batch: True
    hook = ConditionalObjectiveActivation.Factory(named_conditions={"value_loss": condition})()

    assert isinstance(hook, ConditionalObjectiveActivation)
    assert hook.named_conditions == {"value_loss": condition}


def test_hook_factory_get_hook_type_rejects_factory_without_declared_type():
    class MissingTypeFactory(HookFactory):
        pass

    with pytest.raises(NotImplementedError, match="must implement 'get_hook_type'"):
        MissingTypeFactory.get_hook_type()
