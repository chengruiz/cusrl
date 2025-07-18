import numpy as np
import torch

import cusrl


def test_mlp_inference():
    module = cusrl.Mlp.Factory([256, 128])(42, 12).inference()

    input_tensor = torch.randn(10, 42)
    output_tensor1 = module(input_tensor)
    output_tensor2 = module(input_tensor)
    assert isinstance(output_tensor1, torch.Tensor) and output_tensor1.shape == (10, 12)
    assert isinstance(output_tensor2, torch.Tensor) and output_tensor2.shape == (10, 12)
    assert torch.allclose(output_tensor1, output_tensor2)

    input_numpy = input_tensor.cpu().numpy()
    output_numpy1 = module(input_numpy)
    output_numpy2 = module(input_numpy)
    assert isinstance(output_numpy1, np.ndarray) and output_numpy1.shape == (10, 12)
    assert isinstance(output_numpy2, np.ndarray) and output_numpy2.shape == (10, 12)
    assert np.allclose(output_numpy1, output_numpy2)


def test_lstm_inference():
    module = cusrl.Rnn.Factory("LSTM", hidden_size=256, num_layers=2)(42, 12).inference()

    input_tensor = torch.randn(10, 42)
    output_tensor1 = module(input_tensor)
    output_tensor2 = module(input_tensor)
    assert isinstance(output_tensor1, torch.Tensor) and output_tensor1.shape == (10, 12)
    assert isinstance(output_tensor2, torch.Tensor) and output_tensor2.shape == (10, 12)
    assert not torch.allclose(output_tensor1, output_tensor2)

    input_numpy = input_tensor.cpu().numpy()
    output_numpy1 = module(input_numpy)
    output_numpy2 = module(input_numpy)
    assert isinstance(output_numpy1, np.ndarray) and output_numpy1.shape == (10, 12)
    assert isinstance(output_numpy2, np.ndarray) and output_numpy2.shape == (10, 12)
    assert not np.allclose(output_numpy1, output_numpy2)
