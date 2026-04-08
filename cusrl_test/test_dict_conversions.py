from dataclasses import dataclass

import torch

import cusrl
from cusrl.utils import from_dict, to_dict
from cusrl.utils.misc import MISSING
from cusrl.utils.scheduler import LessThan


def double_value(value: int) -> int:
    return value * 2


def triple_value(value: int) -> int:
    return value * 3


@dataclass
class InnerConfig:
    name: str
    enabled: bool
    indices: tuple[int, int]


@dataclass
class ExampleConfig:
    count: int
    ratio: float
    active: bool
    title: str
    note: str | None
    shape: tuple[int, int]
    tags: list[str]
    metadata: dict[str, int]
    region: slice
    device: torch.device
    optimizer_cls: type[torch.optim.Optimizer]
    transform: object
    inner: InnerConfig


def test_from_dict_does_not_mutate_input_dict():
    optimizer_factory = cusrl.OptimizerFactory("AdamW", defaults={"lr": 1e-3})
    data = to_dict(optimizer_factory)

    restored_once = from_dict(optimizer_factory, data)
    restored_twice = from_dict(optimizer_factory, data)

    assert isinstance(restored_once, cusrl.OptimizerFactory)
    assert isinstance(restored_twice, cusrl.OptimizerFactory)
    assert "__class__" in data


def test_from_dict_prefers_dict_values_over_dict_methods():
    data = {"items": 1, "keys": 2}

    restored = from_dict(data, data)

    assert restored["items"] == 1
    assert restored["keys"] == 2


def test_from_dict_removes_multiple_sequence_items():
    restored = from_dict([10, 20, 30], [10, MISSING, MISSING])

    assert restored == [10]


def create_example_config() -> ExampleConfig:
    return ExampleConfig(
        count=3,
        ratio=1.5,
        active=True,
        title="demo",
        note=None,
        shape=(2, 4),
        tags=["alpha", "beta"],
        metadata={"epochs": 10, "layers": 2},
        region=slice(1, 5, 2),
        device=torch.device("cpu"),
        optimizer_cls=torch.optim.AdamW,
        transform=double_value,
        inner=InnerConfig(name="inner", enabled=True, indices=(0, 2)),
    )


def test_from_dict_round_trips_dataclass_fields():
    config = create_example_config()

    round_trip = from_dict(None, to_dict(config))

    assert isinstance(round_trip, ExampleConfig)
    assert round_trip.count == config.count
    assert round_trip.ratio == config.ratio
    assert round_trip.active is config.active
    assert round_trip.title == config.title
    assert round_trip.note is config.note
    assert round_trip.shape == config.shape
    assert round_trip.tags == config.tags
    assert round_trip.metadata == config.metadata
    assert round_trip.region == config.region
    assert round_trip.device == config.device
    assert round_trip.optimizer_cls == config.optimizer_cls
    assert round_trip.transform == config.transform
    assert round_trip.inner == config.inner


def test_from_dict_updates_dataclass_fields():
    config = create_example_config()
    modified = to_dict(config)
    modified["count"] = 5
    modified["ratio"] = 2.5
    modified["active"] = False
    modified["title"] = "updated"
    modified["shape"] = (8, 16)
    modified["tags"] = ["gamma", "delta"]
    modified["metadata"]["epochs"] = 20
    modified["region"]["start"] = 2
    modified["device"]["__str__"] = "cuda:0"
    modified["optimizer_cls"] = to_dict(torch.optim.SGD)
    modified["transform"] = to_dict(triple_value)
    modified["inner"]["name"] = "updated_inner"
    modified["inner"]["enabled"] = False
    modified["inner"]["indices"] = (1, 3)

    restored = from_dict(config, modified)

    assert isinstance(restored, ExampleConfig)
    assert restored.count == 5
    assert restored.ratio == 2.5
    assert restored.active is False
    assert restored.title == "updated"
    assert restored.note is None
    assert restored.shape == (8, 16)
    assert restored.tags == ["gamma", "delta"]
    assert restored.metadata == {"epochs": 20, "layers": 2}
    assert restored.region == slice(2, 5, 2)
    assert restored.device == torch.device("cuda:0")
    assert restored.optimizer_cls == torch.optim.SGD
    assert restored.transform == triple_value
    assert restored.inner == InnerConfig(name="updated_inner", enabled=False, indices=(1, 3))


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
            cusrl.hook.AdaptiveLRSchedule(warmup_iterations=100, initial_scale=0.1),
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
            cusrl.hook.PpoSurrogateLoss(),
            cusrl.hook.RandomNetworkDistillation(
                cusrl.Mlp.Factory(hidden_dims=(256, 256)), output_dim=16, reward_scale=0.1
            ),
            cusrl.hook.ReturnPrediction(),
            cusrl.hook.RewardShaping(),
            cusrl.hook.StatePrediction((0, 2, 4)),
            cusrl.hook.SymmetricArchitecture(),
            cusrl.hook.SymmetricDataAugmentation(),
            cusrl.hook.MirrorSymmetryLoss(1.0),
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
