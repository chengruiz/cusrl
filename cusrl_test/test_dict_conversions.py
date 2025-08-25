import torch

import cusrl
from cusrl.utils import from_dict, to_dict
from cusrl.utils.scheduler import LessThan


def test_dict_conversions():
    agent_factory = cusrl.ActorCritic.Factory(
        num_steps_per_update=24,
        actor_factory=cusrl.Actor.Factory(
            backbone_factory=cusrl.Mlp.Factory(
                hidden_dims=(256, 256),
                activation_fn="ReLU",
                ends_with_activation=True,
            ),
            distribution_factory=cusrl.NormalDist.Factory(),
        ),
        critic_factory=cusrl.Value.Factory(
            backbone_factory=cusrl.Lstm.Factory(hidden_size=256),
        ),
        optimizer_factory=cusrl.OptimizerFactory("AdamW", defaults={"lr": 1e-3}, actor={"lr": 1e-4}),
        sampler=cusrl.AutoMiniBatchSampler(
            num_epochs=4,
            num_mini_batches=4,
        ),
        hooks=[
            cusrl.hook.ActionSmoothnessLoss(),
            cusrl.hook.AdaptiveLRSchedule(),
            cusrl.hook.AdvantageNormalization(),
            cusrl.hook.AdvantageReduction(),
            cusrl.hook.AdversarialMotionPrior(cusrl.Mlp.Factory(hidden_dims=(256, 256))),
            cusrl.hook.ConditionalObjectiveActivation(),
            cusrl.hook.EntropyLoss(),
            cusrl.hook.GeneralizedAdvantageEstimation(),
            cusrl.hook.GradientClipping(),
            cusrl.hook.HookActivationSchedule("gradient_clipping", LessThan(100)),
            cusrl.hook.MiniBatchWiseLRSchedule(),
            cusrl.hook.ModuleInitialization(),
            cusrl.hook.NextStatePrediction(slice(8, 16)),
            cusrl.hook.ObservationNormalization(),
            cusrl.hook.OnPolicyBufferCapacitySchedule(lambda i: 32 if i < 100 else 64),
            cusrl.hook.OnPolicyPreparation(),
            cusrl.hook.OnPolicyStatistics(sampler=cusrl.AutoMiniBatchSampler()),
            cusrl.hook.HookParameterSchedule(
                "action_smoothness_loss", "weight_1st_order", lambda i: 0.01 if i < 100 else 0.02
            ),
            cusrl.hook.PPOSurrogateLoss(),
            cusrl.hook.RandomNetworkDistillation(
                cusrl.Mlp.Factory(hidden_dims=(256, 256)), output_dim=16, reward_scale=0.1
            ),
            cusrl.hook.ReturnPrediction(),
            cusrl.hook.RewardShaping(),
            cusrl.hook.StatePrediction((0, 2, 4)),
            cusrl.hook.SymmetricArchitecture(),
            cusrl.hook.SymmetricDataAugmentation(),
            cusrl.hook.SymmetryLoss(1.0),
            cusrl.hook.ThresholdLRSchedule(),
            cusrl.hook.ValueComputation(),
            cusrl.hook.ValueLoss(),
        ],
        device="cuda",
        compile=False,
        autocast=False,
    )

    # Test to_dict conversion
    original_dict = to_dict(agent_factory)

    # Test from_dict with modifications
    # Create a modified version of the dictionary
    modified_dict = original_dict.copy()

    # Modify some values to test from_dict functionality
    modified_dict["num_steps_per_update"] = 48  # Change from 24 to 48
    modified_dict["device"] = "cpu"  # Change from 'cuda' to 'cpu'
    modified_dict["compile"] = True  # Change from False to True

    # Modify nested values
    modified_dict["actor_factory"]["backbone_factory"]["hidden_dims"] = (512, 512)  # Change from (256, 256)
    modified_dict["critic_factory"]["backbone_factory"]["hidden_size"] = 512  # Change from 256
    modified_dict["optimizer_factory"]["defaults"]["lr"] = 0.002  # Change from 0.001
    modified_dict["optimizer_factory"]["cls"] = "<class 'SGD' from 'torch.optim.sgd'>"  # Change from 'AdamW'
    modified_dict["sampler"]["num_epochs"] = 8  # Change from 4

    # Modify hook parameters
    modified_dict["hooks"]["entropy_loss"]["weight"] = 0.02  # Change from 0.01
    modified_dict["hooks"]["generalized_advantage_estimation"]["gamma"] = 0.95  # Change from 0.99
    modified_dict["hooks"]["next_state_prediction"]["target_indices"]["start"] = 10  # Change from 8
    modified_dict["hooks"]["ppo_surrogate_loss"]["clip_ratio"] = 0.3  # Change from 0.2

    # Apply from_dict to create a new agent factory with modifications
    modified_agent_factory = from_dict(agent_factory, modified_dict)

    # Verify the modifications were applied correctly
    assert (
        modified_agent_factory.num_steps_per_update == 48
    ), f"Expected 48, got {modified_agent_factory.num_steps_per_update}"
    assert modified_agent_factory.device == "cpu", f"Expected 'cpu', got {modified_agent_factory.device}"
    assert modified_agent_factory.compile is True, f"Expected True, got {modified_agent_factory.compile}"

    # Verify nested modifications
    assert modified_agent_factory.actor_factory.backbone_factory.hidden_dims == (
        512,
        512,
    ), f"Expected (512, 512), got {modified_agent_factory.actor_factory.backbone_factory.hidden_dims}"
    assert (
        modified_agent_factory.critic_factory.backbone_factory.hidden_size == 512
    ), f"Expected 512, got {modified_agent_factory.critic_factory.backbone_factory.hidden_size}"
    assert (
        modified_agent_factory.optimizer_factory.defaults["lr"] == 0.002
    ), f"Expected 0.002, got {modified_agent_factory.optimizer_factory.defaults['lr']}"
    assert (
        modified_agent_factory.optimizer_factory.cls == torch.optim.SGD
    ), f"Expected <class 'torch.optim.sgd.SGD'>, got {modified_agent_factory.optimizer_factory.cls}"
    assert (
        modified_agent_factory.sampler.num_epochs == 8
    ), f"Expected 8, got {modified_agent_factory.sampler.num_epochs}"

    # Verify hook modifications
    entropy_loss_hook = modified_agent_factory.hooks.entropy_loss
    gae_hook = modified_agent_factory.hooks.generalized_advantage_estimation
    next_state_prediction_hook = modified_agent_factory.hooks.next_state_prediction
    ppo_hook = modified_agent_factory.hooks.ppo_surrogate_loss

    assert entropy_loss_hook.weight == 0.02, f"Expected 0.02, got {entropy_loss_hook.weight}"
    assert gae_hook.gamma == 0.95, f"Expected 0.95, got {gae_hook.gamma}"
    assert (
        next_state_prediction_hook.target_indices.start == 10
    ), f"Expected 10, got {next_state_prediction_hook.target_indices.start}"
    assert ppo_hook.clip_ratio == 0.3, f"Expected 0.3, got {ppo_hook.clip_ratio}"

    # Test round-trip conversion (to_dict -> from_dict should preserve the object)
    round_trip_dict = to_dict(modified_agent_factory)
    round_trip_factory = from_dict(agent_factory, round_trip_dict)

    # Verify round-trip modifications
    assert round_trip_factory.num_steps_per_update == 48, "Round-trip conversion failed"
    assert round_trip_factory.device == "cpu", "Round-trip conversion failed"
