from functools import partial

import torch

from colossalai.fx.tracer.meta_patch import patched_function
from colossalai.testing import clear_cache_before_run


def _run(data, patch_fn):
    try:
        output = patch_fn(data)
        return output
    except Exception as e:
        return e


def _assert_output_shape(data, patch_fn, expect_exception, output_shape):
    output = _run(data, patch_fn)

    if expect_exception:
        assert isinstance(output, AssertionError)
    else:
        assert not isinstance(output, Exception)
        assert output.is_meta
        assert output.shape == output_shape


@clear_cache_before_run()
def test_repeat_interleave():
    patch_fn = patched_function.torch_repeat_interleave

    # examples from https://pytorch.org/docs/stable/generated/torch.repeat_interleave.html
    data = torch.tensor([1, 2, 3])
    materialized_output = torch.repeat_interleave(data, repeats=2)
    repeat_interleave = partial(patch_fn, repeats=2)
    meta_data = data.to("meta")
    _assert_output_shape(
        data=meta_data, patch_fn=repeat_interleave, expect_exception=False, output_shape=materialized_output.shape
    )

    data = torch.tensor([[1, 2], [3, 4]])
    materialized_output = torch.repeat_interleave(data, repeats=3, dim=1)
    repeat_interleave = partial(patch_fn, repeats=3, dim=1)
    meta_data = data.to("meta")
    _assert_output_shape(
        data=meta_data, patch_fn=repeat_interleave, expect_exception=False, output_shape=materialized_output.shape
    )

    data = torch.tensor([[1, 2], [3, 4]])
    materialized_output = torch.repeat_interleave(data, repeats=torch.tensor([1, 2]), dim=-1)
    repeat_interleave = partial(patch_fn, repeats=torch.tensor([1, 2]), dim=-1)
    meta_data = data.to("meta")
    _assert_output_shape(
        data=meta_data, patch_fn=repeat_interleave, expect_exception=False, output_shape=materialized_output.shape
    )

    data = torch.tensor([[1, 2], [3, 4]])
    materialized_output = torch.repeat_interleave(data, repeats=torch.tensor([1, 2]), dim=0)
    repeat_interleave = partial(patch_fn, repeats=[1, 2], dim=0)
    meta_data = data.to("meta")
    _assert_output_shape(
        data=meta_data, patch_fn=repeat_interleave, expect_exception=True, output_shape=materialized_output.shape
    )


@clear_cache_before_run()
def test_torch_max():
    data = torch.rand(4, 3)
    out = torch.max(data)
    patched_out = patched_function.torch_max(data)
    assert out.shape == patched_out.shape

    data = torch.rand(4, 3, 2)
    out, idx = torch.max(data, dim=1)
    patched_out, patched_idx = patched_function.torch_max(data, dim=1)
    assert out.shape == patched_out.shape
    assert idx.shape == patched_idx.shape

    data = torch.rand(4, 3, 2)
    out, idx = torch.max(data, dim=1, keepdim=True)
    patched_out, patched_idx = patched_function.torch_max(data, dim=1, keepdim=True)
    assert out.shape == patched_out.shape
    assert idx.shape == patched_idx.shape
