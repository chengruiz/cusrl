import torch

from cusrl.sampler.random_sampler import RandomSampler, TemporalRandomSampler
from cusrl.template import Buffer


def test_random_sampler_only_draws_from_valid_prefix():
    buffer = Buffer(capacity=4, parallelism=1, device="cpu")
    for step in range(3):
        buffer.push({"observation": torch.tensor([[float(step)]])})

    _, batch = next(iter(RandomSampler(num_batches=1, batch_size=32)(buffer)))

    assert set(batch["observation"].squeeze(-1).tolist()) <= {0.0, 1.0, 2.0}


def test_temporal_random_sampler_trims_unfilled_tail():
    buffer = Buffer(capacity=4, parallelism=1, device="cpu")
    for step in range(3):
        value = torch.tensor([[float(step)]])
        buffer.push({"observation": value, "actor_memory": value + 100})

    metadata, batch = next(iter(TemporalRandomSampler(num_batches=1, batch_size=1)(buffer)))

    assert metadata["temporal"] is True
    assert torch.equal(batch["observation"].squeeze(-1).squeeze(-1), torch.tensor([0.0, 1.0, 2.0]))
    assert torch.equal(batch["actor_memory"].squeeze(-1), torch.tensor([100.0]))


def test_temporal_random_sampler_uses_valid_chronological_window():
    buffer = Buffer(capacity=4, parallelism=1, device="cpu")
    for step in range(6):
        value = torch.tensor([[float(step)]])
        buffer.push({"observation": value, "actor_memory": value + 100})

    metadata, batch = next(iter(TemporalRandomSampler(num_batches=1, batch_size=1)(buffer)))

    assert metadata["temporal"] is True
    assert torch.equal(batch["observation"].squeeze(-1).squeeze(-1), torch.tensor([2.0, 3.0, 4.0, 5.0]))
    assert torch.equal(batch["actor_memory"].squeeze(-1), torch.tensor([102.0]))


def test_temporal_random_sampler_sequence_len_trims_partial_buffer():
    buffer = Buffer(capacity=4, parallelism=1, device="cpu")
    for step in range(3):
        value = torch.tensor([[float(step)]])
        buffer.push({"observation": value, "actor_memory": value + 100})

    _, batch = next(iter(TemporalRandomSampler(num_batches=1, batch_size=1, sequence_len=2)(buffer)))

    sequence = tuple(batch["observation"].squeeze(-1).squeeze(-1).tolist())
    assert sequence in {(0.0, 1.0), (1.0, 2.0)}
    assert batch["actor_memory"].item() == sequence[0] + 100.0


def test_temporal_random_sampler_sequence_len_uses_random_wrapped_window():
    buffer = Buffer(capacity=4, parallelism=1, device="cpu")
    for step in range(6):
        value = torch.tensor([[float(step)]])
        buffer.push({"observation": value, "actor_memory": value + 100})

    _, batch = next(iter(TemporalRandomSampler(num_batches=1, batch_size=1, sequence_len=2)(buffer)))

    sequence = tuple(batch["observation"].squeeze(-1).squeeze(-1).tolist())
    assert sequence in {(2.0, 3.0), (3.0, 4.0), (4.0, 5.0)}
    assert batch["actor_memory"].item() == sequence[0] + 100.0


def test_temporal_random_sampler_samples_start_and_env_per_sample():
    buffer = Buffer(capacity=4, parallelism=2, device="cpu")
    for step in range(6):
        observation = torch.tensor([[float(step)], [float(step + 10)]])
        actor_memory = observation + 100
        buffer.push({"observation": observation, "actor_memory": actor_memory})

    _, batch = next(iter(TemporalRandomSampler(num_batches=1, batch_size=2, sequence_len=2)(buffer)))

    valid_sequences = {
        (2.0, 3.0),
        (3.0, 4.0),
        (4.0, 5.0),
        (12.0, 13.0),
        (13.0, 14.0),
        (14.0, 15.0),
    }
    sampled_sequences = batch["observation"].squeeze(-1).T.tolist()
    sampled_memories = batch["actor_memory"].squeeze(-1).tolist()

    for sequence, memory in zip(sampled_sequences, sampled_memories):
        sequence_tuple = tuple(sequence)
        assert sequence_tuple in valid_sequences
        assert memory == sequence_tuple[0] + 100.0
