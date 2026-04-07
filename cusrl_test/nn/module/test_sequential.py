import pytest
import torch
from torch import nn

import cusrl
from cusrl.nn.module.stub import Identity


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

    rnn_factory = cusrl.Rnn.Factory(module_type="RNN", hidden_size=rnn_hidden_dim)
    mlp_factory = cusrl.Mlp.Factory(hidden_dims=[mlp_hidden_dim], activation_fn=nn.ReLU)
    sequential_factory = cusrl.Sequential.Factory(factories=[rnn_factory, mlp_factory], hidden_dims=[None])
    sequential_module = sequential_factory(input_dim=input_dim, output_dim=output_dim)

    assert sequential_module.is_recurrent
    assert sequential_module.input_dim == input_dim
    assert sequential_module.output_dim == output_dim

    dummy_input_seq = torch.randn(seq_len, batch_size, input_dim)
    output_seq, memory_seq = sequential_module(dummy_input_seq)
    assert output_seq.shape == (seq_len, batch_size, output_dim)
    assert memory_seq is not None
    assert isinstance(memory_seq["0"], torch.Tensor)
    assert memory_seq["0"].shape == (batch_size, rnn_hidden_dim)

    dummy_input_tensor = torch.randn(batch_size, input_dim)
    output_tensor, memory_tensor = sequential_module(dummy_input_tensor)

    assert output_tensor.shape == (batch_size, output_dim)
    assert memory_tensor is not None
    assert isinstance(memory_tensor["0"], torch.Tensor)
    assert memory_tensor["0"].shape == (batch_size, rnn_hidden_dim)

    done_tensor = torch.zeros(batch_size, dtype=torch.bool)
    done_tensor[0] = True
    _, reset_mem = sequential_module(dummy_input_tensor)
    sequential_module.reset_memory(reset_mem, done_tensor)
    assert reset_mem is not None
    assert torch.all(reset_mem["0"][0, :] == 0.0)


def test_sequential_rnn_rnn():
    batch_size = 4
    input_dim = 16
    rnn_hidden_dim1 = 32
    rnn_hidden_dim2 = 24
    output_dim = 8
    seq_len = 10

    lstm_factory = cusrl.Rnn.Factory(module_type="LSTM", hidden_size=rnn_hidden_dim1)
    gru_factory = cusrl.Rnn.Factory(module_type="GRU", hidden_size=rnn_hidden_dim2)
    sequential_factory = cusrl.Sequential.Factory(factories=[lstm_factory, gru_factory], hidden_dims=[None])
    sequential_module = sequential_factory(input_dim=input_dim, output_dim=output_dim)

    assert sequential_module.is_recurrent
    assert sequential_module.input_dim == input_dim
    assert sequential_module.output_dim == output_dim

    dummy_input_seq = torch.randn(seq_len, batch_size, input_dim)
    output_seq, memory_seq = sequential_module(dummy_input_seq)

    assert output_seq.shape == (seq_len, batch_size, output_dim)
    assert memory_seq is not None
    assert memory_seq["0"]["hidden"].shape == (batch_size, rnn_hidden_dim1)
    assert memory_seq["0"]["cell"].shape == (batch_size, rnn_hidden_dim1)
    assert isinstance(memory_seq["1"], torch.Tensor)
    assert memory_seq["1"].shape == (batch_size, rnn_hidden_dim2)

    dummy_input_tensor = torch.randn(batch_size, input_dim)
    output_tensor, memory_tensor = sequential_module(dummy_input_tensor)

    assert output_tensor.shape == (batch_size, output_dim)
    assert memory_tensor is not None
    assert memory_tensor["0"]["hidden"].shape == (batch_size, rnn_hidden_dim1)
    assert memory_tensor["0"]["cell"].shape == (batch_size, rnn_hidden_dim1)
    assert isinstance(memory_tensor["1"], torch.Tensor)
    assert memory_tensor["1"].shape == (batch_size, rnn_hidden_dim2)

    done_tensor = torch.zeros(batch_size, dtype=torch.bool)
    done_tensor[0] = True
    _, reset_mem = sequential_module(dummy_input_tensor)
    sequential_module.reset_memory(reset_mem, done_tensor)

    assert reset_mem is not None
    assert torch.all(reset_mem["0"]["hidden"][0, :] == 0.0)
    assert torch.all(reset_mem["0"]["cell"][0, :] == 0.0)
    assert torch.all(reset_mem["1"][0, :] == 0.0)


def test_sequential_tracks_and_clears_intermediate_representations():
    module = cusrl.Sequential(
        cusrl.Mlp(input_dim=4, hidden_dims=[6]),
        Identity(input_dim=6, output_dim=None),
    )

    output = module(torch.randn(2, 4))

    assert output.shape == (2, 6)
    assert "0/Mlp.output" in module.intermediate_repr
    assert "1/Identity.output" in module.intermediate_repr

    module.clear_intermediate_repr()
    assert module.intermediate_repr == {}


def test_sequential_factory_requires_hidden_dims_for_all_intermediate_layers():
    with pytest.raises(ValueError):
        cusrl.Sequential.Factory(
            factories=[cusrl.Mlp.Factory([4]), cusrl.Mlp.Factory([4])],
            hidden_dims=[],
        )(input_dim=4, output_dim=2)
