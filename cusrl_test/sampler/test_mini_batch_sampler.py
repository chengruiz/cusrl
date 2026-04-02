import pytest
import torch

from cusrl.sampler.mini_batch_sampler import MiniBatchSampler, TemporalMiniBatchSampler
from cusrl.template import Buffer


def test_mini_batch_sampler_samples_flattened_transitions():
    buffer = Buffer(capacity=2, parallelism=2, device="cpu")
    for step in range(2):
        buffer.push({"observation": torch.tensor([[float(step * 2)], [float(step * 2 + 1)]])})

    batches = list(MiniBatchSampler(num_epochs=1, num_mini_batches=2, shuffle=False)(buffer))

    first_metadata, first_batch = batches[0]
    second_metadata, second_batch = batches[1]

    assert first_metadata == {
        "epoch_index": 0,
        "mini_batch_index": 0,
        "total_epochs": 1,
        "total_mini_batches": 2,
        "temporal": False,
    }
    assert second_metadata == {
        "epoch_index": 0,
        "mini_batch_index": 1,
        "total_epochs": 1,
        "total_mini_batches": 2,
        "temporal": False,
    }
    sampled = torch.cat([first_batch["observation"], second_batch["observation"]], dim=0).squeeze(-1)
    assert sampled.numel() == 4
    assert torch.equal(sampled.sort().values, torch.tensor([0.0, 1.0, 2.0, 3.0]))


def test_mini_batch_sampler_requires_full_buffer():
    buffer = Buffer(capacity=2, parallelism=1, device="cpu")
    buffer.push({"observation": torch.tensor([[0.0]])})

    with pytest.raises(RuntimeError, match="full buffer"):
        next(iter(MiniBatchSampler()(buffer)))


def test_temporal_mini_batch_sampler_samples_sequences_and_memories():
    buffer = Buffer(capacity=3, parallelism=2, device="cpu")
    for step in range(3):
        observation = torch.tensor([[float(step)], [float(step + 10)]])
        buffer.push({"observation": observation, "actor_memory": observation + 100})

    metadata, batch = next(iter(TemporalMiniBatchSampler(num_epochs=1, num_mini_batches=1, shuffle=False)(buffer)))

    assert metadata == {
        "epoch_index": 0,
        "mini_batch_index": 0,
        "total_epochs": 1,
        "total_mini_batches": 1,
        "temporal": True,
    }
    sampled_sequences = {tuple(sequence.tolist()) for sequence in batch["observation"].squeeze(-1).T}
    assert sampled_sequences == {(0.0, 1.0, 2.0), (10.0, 11.0, 12.0)}
    assert set(batch["actor_memory"].squeeze(-1).tolist()) == {100.0, 110.0}
