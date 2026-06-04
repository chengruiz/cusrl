from cusrl.cli import list_experiments


def test_list_experiments_prints_sorted_registry(monkeypatch, capsys):
    loaded = []
    monkeypatch.setattr(list_experiments, "registry", {"z_env_ppo": object(), "a_env_ppo": object()})
    monkeypatch.setattr(list_experiments, "load_experiment_modules", lambda: loaded.append(True))

    list_experiments.main([])

    assert loaded == [True]
    assert capsys.readouterr().out == "Available experiments:\n- a_env_ppo\n- z_env_ppo\n"


def test_list_experiments_parse_args_keeps_module_remainder():
    args = list_experiments.parse_args(["--module", "custom.module", "--flag", "value"])

    assert args.module == ["custom.module", "--flag", "value"]
    assert args.script is None
