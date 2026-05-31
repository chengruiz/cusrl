from argparse import ArgumentTypeError, Namespace
from types import SimpleNamespace

import pytest

from cusrl.utils import cli_utils


def test_apply_inherited_tyro_args_prepends_filtered_training_args():
    trial = SimpleNamespace(
        metadata={
            "tyro_args": [
                "--name",
                "train-run",
                "--agent.num-steps-per-update",
                "48",
                "--agent.actor-hidden-dims",
                "256",
                "128",
                "--env.device",
                "cuda:0",
                "--env_args",
                "--device cuda:0",
                "--env_kwargs",
                "{}",
                "--num-iterations",
                "20000",
                "--agent.normalize-observation=True",
                "--env-args",
                "--device cpu",
                "--env-kwargs",
                "{}",
                "--agent.lr",
                "0.001",
                "--logger",
                "wandb",
            ],
        },
    )

    extra_args = cli_utils.apply_inherited_tyro_args(
        trial,
        Namespace(inherit_args=True),
        ["--agent.lr", "0.0003"],
    )

    assert extra_args == [
        "--agent.num-steps-per-update",
        "48",
        "--agent.actor-hidden-dims",
        "256",
        "128",
        "--env.device",
        "cuda:0",
        "--env_args",
        "--device cuda:0",
        "--env_kwargs",
        "{}",
        "--agent.normalize-observation=True",
        "--env-args",
        "--device cpu",
        "--env-kwargs",
        "{}",
        "--agent.lr",
        "0.001",
        "--agent.lr",
        "0.0003",
    ]


def test_apply_inherited_tyro_args_can_be_disabled():
    trial = SimpleNamespace(metadata={"tyro_args": ["--agent.num-steps-per-update", "48"]})

    extra_args = cli_utils.apply_inherited_tyro_args(
        trial,
        Namespace(inherit_args=False),
        ["--agent.num-steps-per-update", "24"],
    )

    assert extra_args == ["--agent.num-steps-per-update", "24"]


def test_apply_inherited_tyro_args_noops_without_trial_or_metadata_args():
    extra_args = cli_utils.apply_inherited_tyro_args(
        None,
        Namespace(inherit_args=True),
        ["--agent.device", "cpu"],
    )
    assert extra_args == [
        "--agent.device",
        "cpu",
    ]

    trial = SimpleNamespace(metadata={})
    extra_args = cli_utils.apply_inherited_tyro_args(
        trial,
        Namespace(inherit_args=True),
        ["--agent.device", "cpu"],
    )
    assert extra_args == [
        "--agent.device",
        "cpu",
    ]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("True", True),
        ("false", False),
        ("1", True),
        ("0", False),
        ("yes", True),
        ("no", False),
    ],
)
def test_parse_bool(value, expected):
    assert cli_utils.parse_bool(value) is expected


def test_parse_bool_rejects_invalid_value():
    with pytest.raises(ArgumentTypeError):
        cli_utils.parse_bool("maybe")
