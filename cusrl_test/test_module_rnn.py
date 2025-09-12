import pytest
import torch

import cusrl
from cusrl_test import test_module_consistency


def test_rnn_multi_batch():
    observation_dim = 10
    hidden_size = 32
    num_seqs = 8
    seq_len = 16
    repeat = 3

    input = torch.randn(seq_len, repeat * num_seqs, observation_dim)
    rnn = cusrl.Lstm(input_size=observation_dim, num_layers=2, hidden_size=hidden_size)
    output1, memory1 = rnn(input)

    input_reshaped = input.view(seq_len, repeat, num_seqs, observation_dim)
    output2, memory2 = rnn(input_reshaped)
    assert torch.allclose(output1, output2.flatten(1, -2))
    assert all(torch.allclose(m1, m2.flatten(1, -2)) for m1, m2 in zip(memory1, memory2))

    done = torch.rand(seq_len, num_seqs, 1) < 0.1
    done_repeat = done.repeat(1, repeat, 1)
    output1, _ = rnn(input, memory=memory1, done=done_repeat)
    output2, _ = rnn(input_reshaped, memory=memory2, done=done)
    assert torch.allclose(output1, output2.flatten(1, -2), atol=1e-5)


def test_rnn_consistency():
    input_dim = 10
    hidden_size = 32
    num_seqs = 8
    seq_len = 16

    rnn = cusrl.Lstm(num_layers=2, hidden_size=hidden_size, input_size=input_dim)
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


def test_step_memory():
    input_dim = 10
    hidden_size = 32
    num_seqs = 8
    seq_len = 16

    rnn = cusrl.Actor.Factory(
        cusrl.Gru.Factory(num_layers=2, hidden_size=hidden_size),
        cusrl.NormalDist.Factory(),
    )(input_dim, 12)

    observation = torch.randn(seq_len, num_seqs, input_dim)
    memory1 = memory2 = None

    for i in range(seq_len):
        _, memory1 = rnn(observation[i], memory=memory1)
        memory2 = rnn.step_memory(observation[i], memory=memory2)
        assert torch.allclose(memory1, memory2), "RNN memories are not consistent"
