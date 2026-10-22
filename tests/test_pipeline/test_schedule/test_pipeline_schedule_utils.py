import torch

from colossalai.pipeline.schedule._utils import (
    collect_gradients,
    flatten_tensors_and_grads,
    get_batch_size,
    get_micro_batch,
    merge_batch,
)


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


def test_collect_nested_gradients():
    hidden_states = torch.rand(2, requires_grad=True)
    router_logits = (torch.rand(2, requires_grad=True), torch.rand(2, requires_grad=True))
    outputs = {"hidden_states": hidden_states, "past_router_logits": router_logits}

    (hidden_states.sum() + sum(logits.sum() for logits in router_logits)).backward()

    gradients = collect_gradients(outputs)
    assert gradients is not None
    pairs = flatten_tensors_and_grads(outputs, gradients)
    assert len(pairs) == 3
    for tensor, grad in pairs:
        assert grad is tensor.grad

    assert collect_gradients({"metadata": ("not a tensor",)}) is None


def test_nested_gradient_structure_mismatch():
    output = {"hidden": torch.ones(2), "router": (torch.ones(1),)}
    gradients = {"hidden": torch.ones(2), "router": torch.ones(1)}
    try:
        flatten_tensors_and_grads(output, gradients)
    except ValueError as error:
        assert "structures must match" in str(error)
    else:
        raise AssertionError("Expected mismatched structures to fail")


def test_nested_gradient_filters_none_and_metadata():
    tensor = torch.ones(2, requires_grad=True)
    (tensor * 2).sum().backward()
    output = {"tensor": tensor, "metadata": "x", "unused": torch.ones(1)}
    gradients = collect_gradients(output)
    assert gradients == {"tensor": tensor.grad, "metadata": None, "unused": None}
    assert flatten_tensors_and_grads(output, gradients) == [(tensor, tensor.grad)]
