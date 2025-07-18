import pytest
import torch

import cusrl
from cusrl_test import test_module_consistency


def test_rnn_consistency():
    input_dim = 10
    hidden_size = 32
    num_seqs = 20
    seq_len = 30

    rnn = cusrl.Rnn.Factory("LSTM", num_layers=2, hidden_size=hidden_size)(input_dim=input_dim)
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
    input_dim = 10
    hidden_size = 24
    num_seqs = 20
    seq_len = 30
    action_dim = 5

    rnn = cusrl.Actor.Factory(
        backbone_factory=cusrl.Rnn.Factory("lstm", num_layers=2, hidden_size=hidden_size),
        distribution_factory=cusrl.NormalDist.Factory(),
    )(input_dim, action_dim)
    input = torch.randn(seq_len, num_seqs, input_dim)
    done = torch.rand(seq_len, num_seqs, 1) > 0.8
    _, memory = rnn(input)

    output1 = torch.zeros(seq_len, num_seqs, action_dim)
    backbone_output1 = torch.zeros(seq_len, num_seqs, hidden_size)
    memory1 = memory
    for i in range(seq_len):
        (output, _), memory1 = rnn(input[i], memory=memory1)
        rnn.reset_memory(memory1, done=done[i])
        output1[i] = output
        backbone_output1[i] = rnn.intermediate_repr["backbone.output"]

    (output2, _), _ = rnn(input, memory=memory, done=done)
    backbone_output2 = rnn.intermediate_repr["backbone.output"]
    assert torch.allclose(
        backbone_output1, backbone_output2, atol=1e-5
    ), "Backbone outputs of RNN actor are not consistent"
    assert torch.allclose(output1, output2, atol=1e-5), "Outputs for RNN actor are not consistent"


@pytest.mark.parametrize("rnn_type", ["GRU", "LSTM"])
def test_consistency_during_training(rnn_type):
    test_module_consistency(
        cusrl.Rnn.Factory(rnn_type, num_layers=2, hidden_size=32),
        is_recurrent=True,
    )


def test_step_memory():
    input_dim = 10
    hidden_size = 32
    num_seqs = 20
    seq_len = 30

    rnn = cusrl.Actor.Factory(
        cusrl.Rnn.Factory("GRU", num_layers=2, hidden_size=hidden_size),
        cusrl.NormalDist.Factory(),
    )(input_dim, 12)

    input = torch.randn(seq_len, num_seqs, input_dim)
    memory1 = memory2 = None

    for i in range(seq_len):
        _, memory1 = rnn(input[i], memory=memory1)
        memory2 = rnn.step_memory(input[i], memory=memory2)
        assert torch.allclose(memory1, memory2), "RNN memories are not consistent"
