import numpy as np
import torch

import cusrl


def test_mlp_inference_preserves_tensor_and_numpy_formats():
    module = cusrl.Mlp.Factory([256, 128])(42, 12).inference()
    input_tensor = torch.randn(10, 42)

    output_tensor1 = module(input_tensor)
    output_tensor2 = module(input_tensor)

    assert isinstance(output_tensor1, torch.Tensor)
    assert output_tensor1.shape == (10, 12)
    assert torch.allclose(output_tensor1, output_tensor2)

    input_numpy = input_tensor.cpu().numpy()
    output_numpy1 = module(input_numpy)
    output_numpy2 = module(input_numpy)

    assert isinstance(output_numpy1, np.ndarray)
    assert output_numpy1.shape == (10, 12)
    assert np.allclose(output_numpy1, output_numpy2)


def test_recurrent_inference_updates_memory_and_reset_restores_initial_behavior():
    module = cusrl.Rnn.Factory("LSTM", hidden_size=32, num_layers=2)(42, 12).inference()
    input_tensor = torch.randn(10, 42)

    output1 = module(input_tensor)
    output2 = module(input_tensor)
    module.reset()
    output3 = module(input_tensor)

    assert isinstance(output1, torch.Tensor)
    assert output1.shape == (10, 12)
    assert not torch.allclose(output1, output2)
    assert torch.allclose(output1, output3)


def test_inference_wrapper_keeps_1d_input_unbatched():
    module = cusrl.Mlp.Factory([8])(4, 3).inference()

    output = module(torch.randn(4))

    assert output.shape == (3,)
