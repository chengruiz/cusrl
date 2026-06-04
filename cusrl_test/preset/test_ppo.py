import pytest

import cusrl
from cusrl.preset.ppo import get_distribution_factory, ppo_hook_suite


def test_get_distribution_factory_supports_continuous_and_discrete_action_spaces():
    assert isinstance(get_distribution_factory("continuous", init_std=0.5), cusrl.NormalDist.Factory)
    assert isinstance(get_distribution_factory("discrete"), cusrl.OneHotCategoricalDist.Factory)

    with pytest.raises(ValueError, match="Unsupported action space type"):
        get_distribution_factory("hybrid")


def test_ppo_hook_suite_filters_disabled_hooks_and_includes_enabled_options():
    hooks = ppo_hook_suite(
        normalize_observation=True,
        desired_kl_divergence=0.01,
        max_kl_divergence=0.02,
        empty_cuda_cache=True,
    )
    hook_types = {type(hook) for hook in hooks}

    assert cusrl.hook.ObservationNormalization in hook_types
    assert cusrl.hook.AdaptiveLRSchedule in hook_types
    assert cusrl.hook.EmptyCudaCache in hook_types
    assert all(hook is not None for hook in hooks)


def test_ppo_hook_suite_omits_optional_hooks_when_disabled():
    hooks = ppo_hook_suite(normalize_observation=False, desired_kl_divergence=None, empty_cuda_cache=False)
    hook_types = {type(hook) for hook in hooks}

    assert cusrl.hook.ObservationNormalization not in hook_types
    assert cusrl.hook.AdaptiveLRSchedule not in hook_types
    assert cusrl.hook.EmptyCudaCache not in hook_types
