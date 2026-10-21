"""Strict logical-parameter comparisons for small HF5 acceptance tests."""

import torch
import torch.distributed as dist


def compare_tensor(reference, actual, *, atol=0.005, rtol=0.005):
    assert reference.shape == actual.shape
    assert torch.isfinite(reference).all()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), reference.float(), atol=atol, rtol=rtol)
    return (actual.float() - reference.float()).abs().max().item()


def compare_parameters(reference, actual, *, group=None, shard_dims=None, gradients=True):
    """Compare every named parameter using an explicit partition inventory.

    ``shard_dims`` maps parameter names to their actual partition dimension.
    A shape mismatch without this inventory is an error, not an inferred shard.
    ZeRO-owned gradients require a separate working-buffer inventory; this
    helper must not silently accept cleared .grad as full-gradient coverage.
    """
    expected = dict(reference.named_parameters())
    candidate = dict(actual.named_parameters())
    assert expected.keys() == candidate.keys(), (expected.keys() - candidate.keys(), candidate.keys() - expected.keys())
    shard_dims = shard_dims or {}
    assert shard_dims.keys() <= expected.keys()
    errors = {}
    for name, parameter in expected.items():
        other = candidate[name]
        assert parameter.requires_grad == other.requires_grad, name
        r = parameter.grad if gradients else parameter.detach()
        a = other.grad if gradients else other.detach()
        assert (r is None) == (a is None), name
        if r is None:
            continue
        if name in shard_dims:
            assert group is not None and dist.is_initialized(), name
            parts = [torch.empty_like(a) for _ in range(dist.get_world_size(group))]
            dist.all_gather(parts, a, group=group)
            a = torch.cat(parts, dim=shard_dims[name])
        # Vocabulary padding must be accounted for explicitly by the caller.
        assert r.shape == a.shape, name
        errors[name] = compare_tensor(r, a)
    assert errors, "No parameters compared"
    return errors
