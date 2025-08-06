import cusrl
from cusrl_test import create_dummy_env


def test_smoothness():
    environment = create_dummy_env()
    agent_factory = cusrl.preset.ppo.RecurrentAgentFactory()
    agent_factory.register_hook(
        cusrl.hook.ActionSmoothnessLoss(
            weight_1st_order=0.01,
        ).name_("SmoothnessLossOrder1"),
        after="PpoSurrogateLoss",
    ).register_hook(
        cusrl.hook.ActionSmoothnessLoss(
            weight_2nd_order=[0.01] * environment.action_dim,
        ).name_("SmoothnessLossOrder2"),
        after="SmoothnessLossOrder1",
    )
    assert (
        agent_factory.get_hook_index("SmoothnessLossOrder1") == agent_factory.get_hook_index("SmoothnessLossOrder2") - 1
    )
    cusrl.Trainer(environment, agent_factory, num_iterations=5).run_training_loop()
