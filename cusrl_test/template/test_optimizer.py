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


def test_optimizer_factory_rejects_empty_prefix():
    with pytest.raises(ValueError, match="Empty prefixes"):
        OptimizerFactory("SGD", defaults={"lr": 0.1}, param_groups={"": {"lr": 0.2}})

    with pytest.raises(ValueError, match="Empty prefixes"):
        OptimizerFactory("SGD", defaults={"lr": 0.1}, param_filter=("",))


def test_optimizer_factory_keyword_groups_override_and_sort_prefixes():
    factory = OptimizerFactory(
        torch.optim.SGD,
        defaults={"lr": 0.1},
        param_groups={"actor": {"lr": 0.01}, "actor.backbone": {"lr": 0.001}},
        actor={"lr": 0.02},
    )

    assert list(factory.param_groups) == ["actor.backbone", "actor"]
    assert factory.param_groups["actor"] == {"lr": 0.02}


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
            "actor": OptimizerFactory("SGD", defaults={"lr": 0.1}, param_filter=("actor",)),
            "critic": OptimizerFactory("AdamW", defaults={"lr": 0.01}, param_filter=("critic",)),
        },
        model.named_parameters(),
    )

    assert isinstance(optimizer, OptimizerCollection)
    assert isinstance(optimizer.optimizers["actor"], torch.optim.SGD)
    assert isinstance(optimizer.optimizers["critic"], torch.optim.AdamW)
    assert {group["optimizer_name"] for group in optimizer.param_groups} == {"actor", "critic"}

    group_by_name = {name: group for group in optimizer.param_groups for name in group["param_names"]}
    assert set(group_by_name) == {
        "actor.backbone.weight",
        "actor.backbone.bias",
        "actor.head.weight",
        "actor.head.bias",
        "critic.weight",
        "critic.bias",
    }
    assert group_by_name["actor.head.weight"]["lr"] == pytest.approx(0.1)
    assert group_by_name["critic.weight"]["lr"] == pytest.approx(0.01)


def test_optimizer_collection_zero_grad_step_and_state_dict():
    model = ToyModel()
    optimizer = build_optimizer(
        {
            "actor": OptimizerFactory("SGD", defaults={"lr": 0.1}, param_filter=("actor.head",)),
            "critic": OptimizerFactory("AdamW", defaults={"lr": 0.01}, param_filter=("critic",)),
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


def test_optimizer_collection_rejects_duplicate_parameters():
    model = ToyModel()

    with pytest.raises(ValueError, match="multiple optimizers"):
        build_optimizer(
            {
                "actor": OptimizerFactory("SGD", defaults={"lr": 0.1}, param_filter=("actor",)),
                "actor_head": OptimizerFactory("Adam", defaults={"lr": 0.01}, param_filter=("actor.head",)),
            },
            model.named_parameters(),
        )


def test_optimizer_collection_rejects_empty_names():
    model = ToyModel()

    with pytest.raises(ValueError, match="non-empty"):
        build_optimizer(
            {"": OptimizerFactory("SGD", defaults={"lr": 0.1}, param_filter=("actor",))},
            model.named_parameters(),
        )


def test_optimizer_factory_builds_expected_param_groups():
    model = ToyModel()
    factory = OptimizerFactory(
        "SGD",
        defaults={"lr": 0.1, "momentum": 0.9},
        param_groups={
            "actor": {"lr": 0.01},
            "actor.backbone": {"lr": 0.001},
            "special": {"lr": 0.2},
            "unused": {"lr": 0.3},
        },
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
        param_groups={"actor": {"lr": 0.01}, "actor.backbone": {"lr": 0.001}},
        param_filter=("actor",),
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


def test_optimizer_factory_rejects_empty_filtered_parameter_set():
    model = ToyModel()

    with pytest.raises(ValueError, match="optimizer filter"):
        OptimizerFactory("SGD", defaults={"lr": 0.1}, param_filter=("missing",))(model.named_parameters())


def test_optimizer_factory_round_trips_param_filter():
    factory = OptimizerFactory("Adam", defaults={"lr": 1e-3}, param_filter=("actor", "actor.backbone"))

    restored = from_dict(None, to_dict(factory))

    assert isinstance(restored, OptimizerFactory)
    assert restored.param_filter == ("actor.backbone", "actor")
