import pytest
import torch
from torch import nn

from cusrl.template.optimizer import OptimizerCollection, OptimizerFactory, build_optimizer
from cusrl.utils import from_dict, to_dict


class ToyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.special = nn.Parameter(torch.ones(1))
        self.frozen = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.actor = ToyActor()
        self.actor_extra = nn.Linear(4, 1)
        self.critic = nn.Linear(4, 1)


def test_optimizer_factory_rejects_invalid_param_filter():
    with pytest.raises(TypeError, match="param_filter"):
        OptimizerFactory("SGD", defaults={"lr": 0.1}, param_filter=(object(),))

    with pytest.raises(ValueError, match="Empty prefixes"):
        OptimizerFactory("SGD", defaults={"lr": 0.1}, param_filter=("",))


def test_optimizer_factory_keeps_group_override_order():
    factory = OptimizerFactory(
        torch.optim.SGD,
        defaults={"lr": 0.1},
        group_overrides=[
            ("actor", {"lr": 0.02}),
            ("actor.backbone", {"lr": 0.001}),
        ],
    )

    assert factory.group_overrides == (
        ("actor", {"lr": 0.02}),
        ("actor.backbone", {"lr": 0.001}),
    )


def test_optimizer_factory_accepts_optimizer_class():
    parameter = nn.Parameter(torch.ones(1))
    optimizer = OptimizerFactory(torch.optim.AdamW, defaults={"lr": 1e-3})([("weight", parameter)])

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["param_names"] == ["weight"]


def test_build_optimizer_keeps_single_factory_behavior():
    model = ToyModel()

    optimizer = build_optimizer(OptimizerFactory("SGD", defaults={"lr": 0.1}), model.named_parameters())

    assert isinstance(optimizer, torch.optim.SGD)
    assert not isinstance(optimizer, OptimizerCollection)


def test_build_optimizer_creates_named_optimizer_collection():
    model = ToyModel()

    optimizer = build_optimizer(
        {
            "actor": OptimizerFactory("SGD", defaults={"lr": 0.1}, param_filter="actor"),
            "critic": OptimizerFactory("AdamW", defaults={"lr": 0.01}, param_filter="critic"),
            "rest": OptimizerFactory("Adam", defaults={"lr": 0.001}),
        },
        model.named_parameters(),
    )

    assert isinstance(optimizer, OptimizerCollection)
    assert isinstance(optimizer.optimizers["actor"], torch.optim.SGD)
    assert isinstance(optimizer.optimizers["critic"], torch.optim.AdamW)
    assert isinstance(optimizer.optimizers["rest"], torch.optim.Adam)
    assert {group["optimizer_name"] for group in optimizer.param_groups} == {"actor", "critic", "rest"}

    group_by_name = {name: group for group in optimizer.param_groups for name in group["param_names"]}
    assert set(group_by_name) == {
        "special",
        "actor.backbone.weight",
        "actor.backbone.bias",
        "actor.head.weight",
        "actor.head.bias",
        "actor_extra.weight",
        "actor_extra.bias",
        "critic.weight",
        "critic.bias",
    }
    assert group_by_name["actor.head.weight"]["lr"] == pytest.approx(0.1)
    assert group_by_name["critic.weight"]["lr"] == pytest.approx(0.01)
    assert group_by_name["actor_extra.weight"]["lr"] == pytest.approx(0.001)


def test_optimizer_collection_zero_grad_step_and_state_dict():
    model = ToyModel()
    optimizer = build_optimizer(
        {
            "actor": OptimizerFactory("SGD", defaults={"lr": 0.1}, param_filter="actor.head"),
            "critic": OptimizerFactory("AdamW", defaults={"lr": 0.01}, param_filter="critic"),
            "rest": OptimizerFactory("Adam", defaults={"lr": 0.001}),
        },
        model.named_parameters(),
    )
    assert isinstance(optimizer, OptimizerCollection)

    loss = model.actor.head.weight.sum() + model.critic.weight.sum()
    loss.backward()
    assert model.actor.head.weight.grad is not None
    assert model.critic.weight.grad is not None

    optimizer.zero_grad(set_to_none=True)
    assert model.actor.head.weight.grad is None
    assert model.critic.weight.grad is None

    loss = model.actor.head.weight.sum() + model.critic.weight.sum()
    loss.backward()
    actor_weight = model.actor.head.weight.detach().clone()
    critic_weight = model.critic.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(model.actor.head.weight, actor_weight)
    assert not torch.equal(model.critic.weight, critic_weight)

    state_dict = optimizer.state_dict()
    optimizer.optimizers["actor"].param_groups[0]["lr"] = 0.5
    optimizer.load_state_dict(state_dict)
    assert optimizer.optimizers["actor"].param_groups[0]["lr"] == pytest.approx(0.1)


def test_build_optimizer_consumes_parameters_between_factories():
    model = ToyModel()

    optimizer = build_optimizer(
        {
            "actor_head": OptimizerFactory("SGD", defaults={"lr": 0.1}, param_filter="actor.head"),
            "actor": OptimizerFactory("Adam", defaults={"lr": 0.01}, param_filter="actor"),
            "rest": OptimizerFactory("AdamW", defaults={"lr": 0.001}),
        },
        model.named_parameters(),
    )

    assert isinstance(optimizer, OptimizerCollection)
    names_by_optimizer = {
        name: {param_name for param_group in child_optimizer.param_groups for param_name in param_group["param_names"]}
        for name, child_optimizer in optimizer.optimizers.items()
    }

    assert names_by_optimizer["actor_head"] == {"actor.head.weight", "actor.head.bias"}
    assert names_by_optimizer["actor"] == {"actor.backbone.weight", "actor.backbone.bias"}
    assert "actor.head.weight" not in names_by_optimizer["actor"]
    assert names_by_optimizer["rest"] == {
        "special",
        "actor_extra.weight",
        "actor_extra.bias",
        "critic.weight",
        "critic.bias",
    }


def test_build_optimizer_rejects_unassigned_parameters():
    model = ToyModel()

    with pytest.raises(ValueError, match="not assigned"):
        build_optimizer(
            {
                "actor": OptimizerFactory("SGD", defaults={"lr": 0.1}, param_filter="actor"),
                "critic": OptimizerFactory("AdamW", defaults={"lr": 0.01}, param_filter="critic"),
            },
            model.named_parameters(),
        )


def test_build_optimizer_rejects_single_factory_with_unassigned_parameters():
    model = ToyModel()

    with pytest.raises(ValueError, match="not assigned"):
        build_optimizer(OptimizerFactory("SGD", defaults={"lr": 0.1}, param_filter="actor"), model.named_parameters())


def test_optimizer_collection_rejects_empty_names():
    model = ToyModel()

    with pytest.raises(ValueError, match="non-empty"):
        build_optimizer(
            {"": OptimizerFactory("SGD", defaults={"lr": 0.1}, param_filter="actor")},
            model.named_parameters(),
        )


def test_optimizer_factory_builds_expected_param_groups():
    model = ToyModel()
    factory = OptimizerFactory(
        "SGD",
        defaults={"lr": 0.1, "momentum": 0.9},
        group_overrides=[
            ("actor.backbone", {"lr": 0.001}),
            ("actor", {"lr": 0.01}),
            ("special", {"lr": 0.2}),
            ("unused", {"lr": 0.3}),
        ],
    )

    optimizer = factory(model.named_parameters())
    group_by_name = {name: group for group in optimizer.param_groups for name in group["param_names"]}
    expected_names = {name for name, param in model.named_parameters() if param.requires_grad}

    assert isinstance(optimizer, torch.optim.SGD)
    assert set(group_by_name) == expected_names
    assert "frozen" not in group_by_name
    assert {group["lr"] for group in optimizer.param_groups} == {0.001, 0.01, 0.1, 0.2}
    assert all(len(group["params"]) == len(group["param_names"]) for group in optimizer.param_groups)
    assert all(group["momentum"] == pytest.approx(0.9) for group in optimizer.param_groups)

    assert group_by_name["special"]["lr"] == pytest.approx(0.2)
    assert group_by_name["actor.backbone.weight"]["lr"] == pytest.approx(0.001)
    assert group_by_name["actor.backbone.bias"]["lr"] == pytest.approx(0.001)
    assert group_by_name["actor.head.weight"]["lr"] == pytest.approx(0.01)
    assert group_by_name["actor.head.bias"]["lr"] == pytest.approx(0.01)
    assert group_by_name["actor_extra.weight"]["lr"] == pytest.approx(0.1)
    assert group_by_name["actor_extra.bias"]["lr"] == pytest.approx(0.1)
    assert group_by_name["critic.weight"]["lr"] == pytest.approx(0.1)
    assert group_by_name["critic.bias"]["lr"] == pytest.approx(0.1)


def test_optimizer_factory_filters_named_parameters_before_grouping():
    model = ToyModel()
    factory = OptimizerFactory(
        "SGD",
        defaults={"lr": 0.1},
        group_overrides=[
            ("actor.backbone", {"lr": 0.001}),
            ("actor", {"lr": 0.01}),
        ],
        param_filter="actor",
    )

    optimizer = factory(model.named_parameters())
    group_by_name = {name: group for group in optimizer.param_groups for name in group["param_names"]}

    assert set(group_by_name) == {
        "actor.backbone.weight",
        "actor.backbone.bias",
        "actor.head.weight",
        "actor.head.bias",
    }
    assert group_by_name["actor.backbone.weight"]["lr"] == pytest.approx(0.001)
    assert group_by_name["actor.head.weight"]["lr"] == pytest.approx(0.01)


def test_optimizer_factory_accepts_callable_param_filter():
    model = ToyModel()
    factory = OptimizerFactory(
        "SGD",
        defaults={"lr": 0.1},
        param_filter=lambda name, param: param.ndim == 2 and not name.startswith("actor.head"),
    )

    optimizer = factory(model.named_parameters())
    group_by_name = {name: group for group in optimizer.param_groups for name in group["param_names"]}

    assert set(group_by_name) == {
        "actor.backbone.weight",
        "actor_extra.weight",
        "critic.weight",
    }


def test_optimizer_factory_applies_selector_group_overrides():
    model = ToyModel()
    factory = OptimizerFactory(
        "SGD",
        defaults={"lr": 0.1},
        group_overrides=[
            (lambda name, param: param.ndim == 1, {"lr": 0.03}),
            (lambda name, param: name.startswith("actor"), {"lr": 0.02}),
        ],
    )

    optimizer = factory(model.named_parameters())
    group_by_name = {name: group for group in optimizer.param_groups for name in group["param_names"]}

    assert group_by_name["actor.backbone.bias"]["lr"] == pytest.approx(0.03)
    assert group_by_name["critic.bias"]["lr"] == pytest.approx(0.03)
    assert group_by_name["actor.backbone.weight"]["lr"] == pytest.approx(0.02)
    assert group_by_name["actor.head.weight"]["lr"] == pytest.approx(0.02)
    assert group_by_name["critic.weight"]["lr"] == pytest.approx(0.1)


def test_optimizer_factory_uses_first_matching_group_override():
    model = ToyModel()
    factory = OptimizerFactory(
        "SGD",
        defaults={"lr": 0.1},
        group_overrides=[
            (lambda name, param: param.ndim == 1, {"lr": 0.03}),
            ("actor", {"lr": 0.02}),
        ],
    )

    optimizer = factory(model.named_parameters())
    group_by_name = {name: group for group in optimizer.param_groups for name in group["param_names"]}

    assert group_by_name["actor.backbone.bias"]["lr"] == pytest.approx(0.03)
    assert group_by_name["actor.backbone.weight"]["lr"] == pytest.approx(0.02)


def test_optimizer_factory_rejects_empty_filtered_parameter_set():
    model = ToyModel()

    with pytest.raises(ValueError, match="optimizer filter"):
        OptimizerFactory("SGD", defaults={"lr": 0.1}, param_filter=lambda name, param: False)(model.named_parameters())


def test_optimizer_rule_rejects_invalid_selector():
    with pytest.raises(ValueError, match="Empty prefixes"):
        OptimizerFactory("SGD", defaults={"lr": 0.1}, group_overrides=[("", {"lr": 0.1})])

    with pytest.raises(TypeError, match="selector"):
        OptimizerFactory("SGD", defaults={"lr": 0.1}, group_overrides=[(object(), {"lr": 0.1})])


def test_optimizer_factory_round_trips_param_filter():
    factory = OptimizerFactory("Adam", defaults={"lr": 1e-3}, param_filter=("actor", "actor.backbone"))

    restored = from_dict(None, to_dict(factory))

    assert isinstance(restored, OptimizerFactory)
    assert restored.param_filter == ("actor", "actor.backbone")
