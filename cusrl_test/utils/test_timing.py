import time

import pytest
import torch

import cusrl


def test_timer():
    timer = cusrl.utils.Timer()

    @timer.decorate("test")
    def func():
        time.sleep(0.1)

    for _ in range(10):
        func()
    assert 0.99 <= timer["test"] <= 1.01
    timer.clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for CUDA timing")
def test_timer_cuda():
    timer = cusrl.utils.Timer("cuda")

    @timer.decorate("test")
    def func():
        x = torch.randn(2048, 2048, device="cuda")
        y = x @ x
        return y.sum()

    for _ in range(5):
        func()
    assert timer["test"] > 0.0
    timer.clear()


def test_rate():
    rate = cusrl.utils.Rate(10)
    start_time = time.time()
    for _ in range(10):
        rate.tick()
    elapsed_time = time.time() - start_time
    assert 0.99 <= elapsed_time <= 1.01
