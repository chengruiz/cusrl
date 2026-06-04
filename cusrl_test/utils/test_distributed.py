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
