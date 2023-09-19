import torch

from colossalai.pipeline.schedule._utils import get_batch_size, get_micro_batch, merge_batch


def test_get_batch_size():
    tensor = torch.rand(2, 3)
    assert get_batch_size(tensor) == 2
    assert get_batch_size([tensor]) == 2
    assert get_batch_size((1, tensor)) == 2
    assert get_batch_size({"tensor": tensor}) == 2
    assert get_batch_size({"dummy": [1], "tensor": tensor}) == 2
    assert get_batch_size({"tensor": [tensor]}) == 2


def test_get_micro_batch():
    x = torch.rand(2, 1)
    y = torch.rand(2, 3)
    micro_batch = get_micro_batch(x, 0, 1)
    assert torch.equal(micro_batch, x[0:1])
    micro_batch = get_micro_batch(x, 1, 1)
    assert torch.equal(micro_batch, x[1:2])
    micro_batch = get_micro_batch([x, y], 0, 1)
    assert torch.equal(micro_batch[0], x[0:1])
    assert torch.equal(micro_batch[1], y[0:1])
    micro_batch = get_micro_batch([x, y], 1, 1)
    assert torch.equal(micro_batch[0], x[1:2])
    assert torch.equal(micro_batch[1], y[1:2])
    micro_batch = get_micro_batch({"x": x, "y": y}, 0, 1)
    assert torch.equal(micro_batch["x"], x[0:1])
    assert torch.equal(micro_batch["y"], y[0:1])
    micro_batch = get_micro_batch({"x": x, "y": y}, 1, 1)
    assert torch.equal(micro_batch["x"], x[1:2])
    assert torch.equal(micro_batch["y"], y[1:2])


def test_merge_batch():
    x = torch.rand(2, 1)
    y = torch.rand(2, 3)
    merged = merge_batch([x[0:1], x[1:2]])
    assert torch.equal(merged, x)
    merged = merge_batch([[x[0:1], y[0:1]], [x[1:2], y[1:2]]])
    assert torch.equal(merged[0], x)
    assert torch.equal(merged[1], y)
    merged = merge_batch([{"x": x[0:1], "y": y[0:1]}, {"x": x[1:2], "y": y[1:2]}])
    assert torch.equal(merged["x"], x)
    assert torch.equal(merged["y"], y)
