import pytest

from cusrl.cli import benchmark, export, play, train


@pytest.mark.parametrize("module", [play, benchmark, export, train])
def test_inherit_args_defaults_to_true(module):
    args, extra_args = module.parse_args(["--agent.device", "cpu"])

    assert args.inherit_args is True
    assert extra_args == ["--agent.device", "cpu"]


@pytest.mark.parametrize("module", [play, benchmark, export, train])
def test_inherit_args_false_is_parsed_as_command_arg(module):
    args, extra_args = module.parse_args(["--inherit-args", "False", "--agent.device", "cpu"])

    assert args.inherit_args is False
    assert extra_args == ["--agent.device", "cpu"]


def test_train_inherits_args_from_checkpoint(monkeypatch, tmp_path):
    captured = {}
    checkpoint_path = tmp_path / "trial/ckpt/ckpt_3.pt"

    class ExperimentSpec:
        def to_training_factory(self):
            return object()

    class Trainer:
        def run_training_loop(self):
            captured["ran"] = True

    def load_checkpoint_from_args(args):
        captured["checkpoint"] = args.checkpoint
        return type(
            "Trial",
            (),
            {
                "checkpoint_path": checkpoint_path,
                "metadata": {"tyro_args": ["--agent.num-steps-per-update", "48", "--name", "old-run"]},
            },
        )()

    def tyro_cli(*, prog, default, args):
        captured["prog"] = prog
        captured["tyro_args"] = args

        def trainer_factory(checkpoint_path=None, trial_metadata=None):
            captured["trainer_checkpoint_path"] = checkpoint_path
            captured["trial_metadata"] = trial_metadata
            return Trainer()

        return trainer_factory

    monkeypatch.setattr(train.cusrl, "set_global_seed", lambda seed: None)
    monkeypatch.setattr(train.cli_utils, "load_checkpoint_from_args", load_checkpoint_from_args)
    monkeypatch.setattr(train.cli_utils, "load_experiment_spec_from_args", lambda args: ExperimentSpec())
    monkeypatch.setattr(train, "tyro_cli", tyro_cli)

    train.main([
        "--environment",
        "Env",
        "--algorithm",
        "Alg",
        "--checkpoint",
        "run",
        "--agent.lr",
        "0.0003",
    ])

    assert captured["checkpoint"] == "run"
    assert captured["tyro_args"] == [
        "--agent.num-steps-per-update",
        "48",
        "--agent.lr",
        "0.0003",
    ]
    assert captured["trainer_checkpoint_path"] == str(checkpoint_path)
    assert captured["trial_metadata"] == {"tyro_args": captured["tyro_args"]}
    assert captured["ran"] is True
