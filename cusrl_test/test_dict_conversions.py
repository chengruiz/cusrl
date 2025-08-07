import cusrl
from cusrl.utils.helper import to_dict


def test_to_dict():
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
            cusrl.hook.HookActivationSchedule("gradient_clipping", cusrl.hook.schedule.LessThan(100)),
            cusrl.hook.MiniBatchWiseLRSchedule(),
            cusrl.hook.ModuleInitialization(),
            cusrl.hook.NextStatePrediction(slice(8, 16)),
            cusrl.hook.ObservationNormalization(),
            cusrl.hook.OnPolicyBufferCapacitySchedule(lambda i: 32 if i < 100 else 64),
            cusrl.hook.OnPolicyPreparation(),
            cusrl.hook.OnPolicyStatistics(sampler=cusrl.AutoMiniBatchSampler()),
            cusrl.hook.ParameterSchedule(
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

    print(to_dict(agent_factory))
