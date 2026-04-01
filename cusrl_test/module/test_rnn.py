import pytest
import torch

import cusrl
from cusrl.utils.nest import flatten_nested, map_nested
from cusrl_test import test_module_consistency


def assert_memory_allclose(memory1, memory2, *, atol=1e-8, rtol=1e-5):
    flat_memory1 = flatten_nested(memory1)
    flat_memory2 = flatten_nested(memory2)
    assert flat_memory1.keys() == flat_memory2.keys()
    for key in flat_memory1:
        assert torch.allclose(flat_memory1[key], flat_memory2[key], atol=atol, rtol=rtol), f"Memory mismatch at {key}"


def clone_memory(memory):
    if memory is None:
        return None
    return map_nested(torch.clone, memory)


@pytest.mark.parametrize(
    ("rnn_type", "memory_keys"),
    [
        ("RNN", {"hidden"}),
        ("GRU", {"hidden"}),
        ("LSTM", {"hidden", "cell"}),
    ],
)
def test_rnn_factory_memory_structure(rnn_type, memory_keys):
    input_dim = 10
    hidden_size = 32
    output_dim = 7
    num_seqs = 4
    seq_len = 6

    rnn = cusrl.Rnn.Factory(rnn_type, num_layers=2, hidden_size=hidden_size)(input_dim, output_dim)
    output, memory = rnn(torch.randn(seq_len, num_seqs, input_dim))

    assert output.shape == (seq_len, num_seqs, output_dim)
    assert set(memory.keys()) == memory_keys
    assert memory["hidden"].shape == (2, num_seqs, hidden_size)
    if "cell" in memory:
        assert memory["cell"].shape == (2, num_seqs, hidden_size)


def test_rnn_factory_rejects_unknown_module_type():
    with pytest.raises(ValueError, match="Unsupported RNN module class"):
        cusrl.Rnn.Factory("unsupported", hidden_size=32)(10)


def test_rnn_multi_batch():
    observation_dim = 10
    hidden_size = 32
    num_seqs = 8
    seq_len = 16
    repeat = 3

    input = torch.randn(seq_len, repeat * num_seqs, observation_dim)
    rnn = cusrl.Lstm(observation_dim, num_layers=2, hidden_size=hidden_size)
    output1, memory1 = rnn(input)

    input_reshaped = input.view(seq_len, repeat, num_seqs, observation_dim)
    output2, memory2 = rnn(input_reshaped)
    assert torch.allclose(output1, output2.flatten(1, -2))
    assert_memory_allclose(memory1, {key: value.flatten(1, -2) for key, value in memory2.items()})

    done = torch.rand(seq_len, num_seqs, 1) < 0.1
    done_repeat = done.repeat(1, repeat, 1)
    output1, _ = rnn(input, memory=memory1, done=done_repeat)
    output2, _ = rnn(input_reshaped, memory=memory2, done=done)
    assert torch.allclose(output1, output2.flatten(1, -2), atol=1e-5)


@pytest.mark.parametrize("rnn_type", ["RNN", "GRU", "LSTM"])
def test_rnn_pack_sequence_consistency(rnn_type):
    input_dim = 10
    hidden_size = 32
    num_seqs = 8
    seq_len = 16
    warmup_len = 4

    rnn = cusrl.Rnn.Factory(rnn_type, num_layers=2, hidden_size=hidden_size)(input_dim)
    warmup = torch.randn(warmup_len, num_seqs, input_dim)
    observation = torch.randn(seq_len, num_seqs, input_dim)
    done = torch.rand(seq_len, num_seqs, 1) > 0.75

    _, initial_memory = rnn(warmup)

    output_step = torch.zeros(seq_len, num_seqs, hidden_size)
    memory_step = clone_memory(initial_memory)
    for i in range(seq_len):
        output, memory_step = rnn(observation[i], memory=memory_step)
        rnn.reset_memory(memory_step, done=done[i])
        output_step[i] = output

    output_unpacked, unpacked_memory = rnn(observation, memory=clone_memory(initial_memory), done=done)
    output_packed, packed_memory = rnn(
        observation,
        memory=clone_memory(initial_memory),
        done=done,
        pack_sequence=True,
    )

    assert unpacked_memory is None
    assert torch.allclose(output_step, output_unpacked, atol=1e-5)
    assert torch.allclose(output_step, output_packed, atol=1e-5)
    assert_memory_allclose(memory_step, packed_memory, atol=1e-5)


def test_rnn_done_requires_sequential():
    rnn = cusrl.Gru(10, hidden_size=32, num_layers=2)
    done = torch.zeros(1, 4, 1, dtype=torch.bool)

    with pytest.raises(ValueError, match="'done' can be provided only when 'sequential' is True"):
        rnn(torch.randn(4, 10), done=done, sequential=False)


def test_rnn_pack_sequence_requires_3d_input():
    input_dim = 10
    num_seqs = 8
    seq_len = 16
    repeat = 3
    rnn = cusrl.Lstm(input_dim, hidden_size=32, num_layers=2)
    observation = torch.randn(seq_len, repeat, num_seqs, input_dim)
    done = torch.zeros(seq_len, num_seqs, 1, dtype=torch.bool)

    with pytest.raises(ValueError, match="Packed RNN input must be 3D"):
        rnn(observation, done=done, pack_sequence=True)


def test_rnn_consistency():
    input_dim = 10
    hidden_size = 32
    num_seqs = 8
    seq_len = 16

    rnn = cusrl.Lstm(input_dim, num_layers=2, hidden_size=hidden_size)
    input = torch.randn(seq_len, num_seqs, input_dim)
    done = torch.rand(seq_len, num_seqs, 1) > 0.8
    _, memory = rnn(input)

    output1 = torch.zeros(seq_len, num_seqs, hidden_size)
    memory1 = memory
    for i in range(seq_len):
        output, memory1 = rnn(input[i], memory=memory1)
        rnn.reset_memory(memory1, done=done[i])
        output1[i] = output

    output2, _ = rnn(input, memory=memory, done=done)
    assert torch.allclose(output1, output2, atol=1e-5), "RNN outputs are not consistent"


def test_rnn_actor_consistency():
    observation_dim = 10
    hidden_size = 32
    num_seqs = 8
    seq_len = 16
    action_dim = 5

    rnn = cusrl.Actor.Factory(
        backbone_factory=cusrl.Lstm.Factory(num_layers=2, hidden_size=hidden_size),
        distribution_factory=cusrl.NormalDist.Factory(),
    )(observation_dim, action_dim)
    observation = torch.randn(seq_len, num_seqs, observation_dim)
    done = torch.rand(seq_len, num_seqs, 1) > 0.8
    _, init_memory = rnn(observation)

    action_mean1 = torch.zeros(seq_len, num_seqs, action_dim)
    memory1 = init_memory
    for i in range(seq_len):
        action_dist, memory1 = rnn(observation[i], memory=memory1)
        rnn.reset_memory(memory1, done=done[i])
        action_mean1[i] = action_dist["mean"]

    action_dist2, _ = rnn(observation, memory=init_memory, done=done)
    action_mean2 = action_dist2["mean"]
    assert torch.allclose(action_mean1, action_mean2, atol=1e-5), "Action means are not consistent"


@pytest.mark.parametrize("rnn_type", ["GRU", "LSTM"])
def test_consistency_during_training(rnn_type):
    test_module_consistency(
        cusrl.Rnn.Factory(rnn_type, num_layers=2, hidden_size=32),
        is_recurrent=True,
    )


@pytest.mark.parametrize("rnn_type", ["RNN", "GRU", "LSTM"])
def test_step_memory(rnn_type):
    input_dim = 10
    hidden_size = 32
    num_seqs = 8
    seq_len = 16

    rnn = cusrl.Actor.Factory(
        cusrl.Rnn.Factory(rnn_type, num_layers=2, hidden_size=hidden_size),
        cusrl.NormalDist.Factory(),
    )(input_dim, 12)

    observation = torch.randn(seq_len, num_seqs, input_dim)
    memory1 = memory2 = None

    for i in range(seq_len):
        _, memory1 = rnn(observation[i], memory=memory1)
        memory2 = rnn.step_memory(observation[i], memory=memory2)
        assert_memory_allclose(memory1, memory2)
