import torch

from cusrl.utils import distributed


def test_single_process_distributed_helpers_are_local_noops(capsys):
    tensor = torch.tensor([1.0, 2.0])

    assert distributed.enabled() is False
    assert distributed.rank() == 0
    assert distributed.local_rank() == 0
    assert distributed.world_size() == 1
    assert distributed.average_dict({"reward": 2.0}) == {"reward": 2.0}
    assert distributed.gather_obj("payload") == ["payload"]
    assert distributed.gather_tensor(tensor) == [tensor]
    assert torch.equal(distributed.gather_stack(tensor), tensor.unsqueeze(0))
    assert distributed.make_none_obj_list() == []
    assert distributed.reduce_mean_(tensor) is tensor

    distributed.gather_print("hello")
    captured = capsys.readouterr()
    assert captured.out == "hello\n"


def test_reduce_mean_var_supports_scalar_and_multidim_tensors(monkeypatch):
    monkeypatch.setattr(distributed, "configure_distributed", lambda: True)
    monkeypatch.setattr(
        distributed,
        "gather_stack",
        lambda x: torch.stack([x, x + torch.tensor([2.0, 4.0]).view(2, *([1] * (x.ndim - 1)))], dim=0),
    )

    for shape in [(), (3,), (2, 1), (2, 4)]:
        mean = torch.ones(shape)
        var = torch.full(shape, 2.0)
        out_mean, out_var = distributed.reduce_mean_var_(mean, var)

        assert out_mean is mean
        assert out_var is var
        assert mean.shape == shape
        assert var.shape == shape
        assert torch.allclose(mean, torch.full(shape, 2.0))
        assert torch.allclose(var, torch.full(shape, 5.0))
