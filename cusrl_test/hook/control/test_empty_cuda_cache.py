import torch

import cusrl


def test_empty_cuda_cache_only_runs_when_cuda_is_available(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty"))

    cusrl.hook.EmptyCudaCache().post_update()

    assert calls == ["empty"]


def test_empty_cuda_cache_skips_cpu_only_runtime(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty"))

    cusrl.hook.EmptyCudaCache().post_update()

    assert calls == []
