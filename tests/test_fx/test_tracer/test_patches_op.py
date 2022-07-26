import torch
from colossalai.fx.tracer.meta_patch import patched_function


def _run(data, function, patch_fn):
    try:
        output = patch_fn(function, data)
        return output
    except Exception as e:
        return e


def _assert_output_shape(data, function, patch_fn, expect_exception, output_shape):
    output = _run(data, function, patch_fn)

    if expect_exception:
        assert isinstance(output, AssertionError)
    else:
        assert not isinstance(output, Exception)
        assert output.is_meta
        assert output.shape == output_shape


def test_repeat_interleave():
    data = torch.tensor([1, 2, 3])
