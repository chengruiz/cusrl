import torch
from torch import nn

import cusrl


def test_sequential_mlp_mlp():
    batch_size = 4
    input_dim = 16
    hidden_dim1 = 32
    hidden_dim2 = 64
    output_dim = 8

    mlp_factory1 = cusrl.Mlp.Factory(hidden_dims=[hidden_dim1], activation_fn=nn.ReLU, ends_with_activation=True)
    mlp_factory2 = cusrl.Mlp.Factory(hidden_dims=[hidden_dim2], activation_fn=nn.ReLU)

    sequential_factory = cusrl.Sequential.Factory(factories=[mlp_factory1, mlp_factory2], hidden_dims=[None])
    sequential_module = sequential_factory(input_dim=input_dim, output_dim=output_dim)
    assert not sequential_module.is_recurrent

    dummy_input = torch.randn(batch_size, input_dim)
    output = sequential_module(dummy_input)
    assert output.shape == (batch_size, output_dim)


def test_sequential_rnn_mlp():
    batch_size = 4
    input_dim = 16
    rnn_hidden_dim = 32
    mlp_hidden_dim = 24
    output_dim = 8
    seq_len = 10

    rnn_factory = cusrl.Rnn.Factory(module_cls="RNN", hidden_size=rnn_hidden_dim)
    mlp_factory = cusrl.Mlp.Factory(hidden_dims=[mlp_hidden_dim], activation_fn=nn.ReLU)
    sequential_factory = cusrl.Sequential.Factory(factories=[rnn_factory, mlp_factory], hidden_dims=[None])
    sequential_module = sequential_factory(input_dim=input_dim, output_dim=output_dim)

    assert sequential_module.is_recurrent
    assert sequential_module.input_dim == input_dim
    assert sequential_module.output_dim == output_dim

    # Test with sequence input (T, N, C)
    dummy_input_seq = torch.randn(seq_len, batch_size, input_dim)
    output_seq, memory_seq = sequential_module(dummy_input_seq)
    assert output_seq.shape == (seq_len, batch_size, output_dim)
    assert isinstance(memory_seq, tuple) and len(memory_seq) == 1
    assert memory_seq[0].shape == (1, batch_size, rnn_hidden_dim)

    # Test with single tensor input (N, C) -> RNN treats as (1, N, C)
    dummy_input_tensor = torch.randn(batch_size, input_dim)
    output_tensor, memory_tensor = sequential_module(dummy_input_tensor)

    assert output_tensor.shape == (batch_size, output_dim)
    assert memory_tensor is not None and len(memory_tensor) == 1
    assert memory_tensor[0].shape == (1, batch_size, rnn_hidden_dim)

    # Test reset_memory
    done_tensor = torch.zeros(batch_size, dtype=torch.bool)  # (N,)
    done_tensor[0] = True  # Reset for the first item in batch

    _, reset_mem = sequential_module(dummy_input_tensor)  # Get a memory state
    sequential_module.reset_memory(reset_mem, done_tensor)
    assert reset_mem is not None and len(reset_mem) == 1
    assert torch.all(reset_mem[0][:, 0, :] == 0.0)


def test_sequential_rnn_rnn():
    batch_size = 4
    input_dim = 16
    rnn_hidden_dim1 = 32  # LSTM hidden dim
    rnn_hidden_dim2 = 24  # GRU hidden dim
    output_dim = 8
    seq_len = 10

    lstm_factory = cusrl.Rnn.Factory(module_cls="LSTM", hidden_size=rnn_hidden_dim1)
    gru_factory = cusrl.Rnn.Factory(module_cls="GRU", hidden_size=rnn_hidden_dim2)
    sequential_factory = cusrl.Sequential.Factory(factories=[lstm_factory, gru_factory], hidden_dims=[None])
    sequential_module = sequential_factory(input_dim=input_dim, output_dim=output_dim)

    assert sequential_module.is_recurrent
    assert sequential_module.input_dim == input_dim
    assert sequential_module.output_dim == output_dim

    # Test with sequence input (T, N, C)
    dummy_input_seq = torch.randn(seq_len, batch_size, input_dim)
    output_seq, memory_seq = sequential_module(dummy_input_seq)

    assert output_seq.shape == (seq_len, batch_size, output_dim)
    assert isinstance(memory_seq, tuple) and len(memory_seq) == 2
    assert isinstance(memory_seq[0], tuple) and len(memory_seq[0]) == 2
    assert memory_seq[0][0].shape == (1, batch_size, rnn_hidden_dim1)
    assert memory_seq[0][1].shape == (1, batch_size, rnn_hidden_dim1)
    assert memory_seq[1].shape == (1, batch_size, rnn_hidden_dim2)

    # Test with single tensor input (N, C) -> RNN treats as (1, N, C)
    dummy_input_tensor = torch.randn(batch_size, input_dim)
    output_tensor, memory_tensor = sequential_module(dummy_input_tensor)

    assert output_tensor.shape == (batch_size, output_dim)
    assert isinstance(memory_tensor, tuple) and len(memory_tensor) == 2
    assert isinstance(memory_tensor[0], tuple) and len(memory_tensor[0]) == 2
    assert memory_tensor[0][0].shape == (1, batch_size, rnn_hidden_dim1)
    assert memory_tensor[0][1].shape == (1, batch_size, rnn_hidden_dim1)
    assert memory_tensor[1].shape == (1, batch_size, rnn_hidden_dim2)

    # Test reset_memory
    done_tensor = torch.zeros(batch_size, dtype=torch.bool)  # (N,)
    done_tensor[0] = True
    _, reset_mem = sequential_module(dummy_input_tensor)  # Get a memory state
    sequential_module.reset_memory(reset_mem, done_tensor)

    assert reset_mem is not None and len(reset_mem) == 2
    assert isinstance(reset_mem[0], tuple) and len(reset_mem[0]) == 2
    assert torch.all(reset_mem[0][0][:, 0, :] == 0.0)
    assert torch.all(reset_mem[0][1][:, 0, :] == 0.0)
    assert torch.all(reset_mem[1][:, 0, :] == 0.0)
