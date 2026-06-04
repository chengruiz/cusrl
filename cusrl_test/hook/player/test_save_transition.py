from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cusrl.hook.player.save_transition import SaveTransition


def test_save_transition_appends_npz_suffix_and_writes_stacked_arrays(tmp_path):
    hook = SaveTransition(tmp_path / "transitions", keys=("observation", "reward"))
    hook.init(SimpleNamespace(agent=None, environment=None))

    hook.step(0, {"observation": torch.tensor([1.0, 2.0]), "reward": np.array([0.5])})
    hook.step(1, {"observation": torch.tensor([3.0, 4.0]), "reward": np.array([1.5])})
    hook.close()

    data = np.load(tmp_path / "transitions.npz")
    assert data["observation"].shape == (2, 2)
    assert np.allclose(data["reward"], np.array([[0.5], [1.5]]))


def test_save_transition_shards_outputs_by_save_interval(tmp_path):
    hook = SaveTransition(tmp_path / "transitions.npz", keys=("reward",), save_interval=2)
    hook.init(SimpleNamespace(agent=None, environment=None))

    hook.step(0, {"reward": np.array([0.5])})
    hook.step(1, {"reward": np.array([1.5])})
    hook.step(2, {"reward": np.array([2.5])})
    hook.close()

    first = np.load(tmp_path / "transitions_000000.npz")
    second = np.load(tmp_path / "transitions_000001.npz")
    assert np.allclose(first["reward"], np.array([[0.5], [1.5]]))
    assert np.allclose(second["reward"], np.array([[2.5]]))


def test_save_transition_rejects_non_positive_save_interval_on_init(tmp_path):
    hook = SaveTransition(tmp_path / "transitions.npz", save_interval=0)

    with pytest.raises(ValueError, match="save_interval"):
        hook.init(SimpleNamespace(agent=None, environment=None))
