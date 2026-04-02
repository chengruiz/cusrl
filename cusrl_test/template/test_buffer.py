import numpy as np
import pytest
import torch

from cusrl.template import Buffer


def test_buffer_setitem_accepts_numpy_arrays():
    buffer = Buffer(capacity=3, parallelism=2, device="cpu")

    buffer["observation"] = np.arange(6, dtype=np.float32).reshape(3, 2, 1)

    observation = buffer["observation"]
    assert isinstance(observation, torch.Tensor)
    assert observation.shape == (3, 2, 1)
    assert observation.device.type == "cpu"


def test_buffer_rejects_rank_one_step_values():
    buffer = Buffer(capacity=3, parallelism=2, device="cpu")

    with pytest.raises(ValueError, match=r"\[parallelism, \.\.\.\]"):
        buffer.push({"observation": torch.tensor([1.0])})


def test_buffer_rejects_rank_one_full_field_values():
    buffer = Buffer(capacity=3, parallelism=2, device="cpu")

    with pytest.raises(ValueError, match=r"\[capacity, parallelism, \.\.\.\]"):
        buffer["observation"] = torch.tensor([1.0, 2.0, 3.0])


def test_buffer_push_preserves_bool_dtype():
    buffer = Buffer(capacity=2, parallelism=1, device="cpu")

    buffer.push({"terminated": torch.tensor([[True]])})

    assert buffer["terminated"].dtype == torch.bool
