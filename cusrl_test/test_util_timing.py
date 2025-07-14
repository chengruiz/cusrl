import time

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


def test_rate():
    rate = cusrl.utils.Rate(10)
    start_time = time.time()
    for _ in range(10):
        rate.tick()
    elapsed_time = time.time() - start_time
    assert 0.99 <= elapsed_time <= 1.01
