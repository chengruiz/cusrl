import torch

import cusrl


def print_module(module, input: torch.Tensor):
    print(module)
    print(f"input: {input.shape} output: {module(input).shape}\n")


def pad_to(string, length):
    padding = (length - len(string)) // 2 - 1
    return "─" * padding + " " + string + " " + "─" * padding


def test_simba():
    print(pad_to("Simba.Factory()(64)", 60))
    module = cusrl.Simba.Factory()(64)
    print_module(module, torch.randn(8, 64))

    print(pad_to("Simba.Factory()(64, 128)", 60))
    module = cusrl.Simba.Factory()(64, 128)
    print_module(module, torch.randn(8, 64))

    print(pad_to("Simba.Factory(256)(64, 128)", 60))
    module = cusrl.Simba.Factory(256)(64, 128)
    print_module(module, torch.randn(8, 64))

    print(pad_to("Simba.Factory(256)(64, 128)", 60))
    module = cusrl.Simba.Factory(256)(64, 128)
    print_module(module, torch.randn(8, 64))

    print(pad_to("Simba.Factory(256, 2)(64, 128)", 60))
    module = cusrl.Simba.Factory(256, 2)(64, 128)
    print_module(module, torch.randn(8, 64))
