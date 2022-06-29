import torch
from colossalai.fx.tracer.meta_patch import patched_module


def _run(data, module, patch_fn):
    try:
        output = patch_fn(module, data)
        return output
    except Exception as e:
        return e


def _assert_output_shape(data, module, patch_fn, expect_exception, output_shape):
    output = _run(data, module, patch_fn)

    if expect_exception:
        assert isinstance(output, AssertionError)
    else:
        assert not isinstance(output, Exception)
        assert output.is_meta
        assert output.shape == output_shape


def test_linear():
    # test linear patch can produce the meta output with correct shape
    data = torch.rand(2, 4, device='meta')
    module = torch.nn.Linear(4, 2)
    _assert_output_shape(data, module, patched_module.torch_nn_linear, False, torch.Size([2, 2]))

    # Test if the linear patch can catch exception when dimension does not match
    data = torch.rand(2, 2, device='meta')
    _assert_output_shape(data, module, patched_module.torch_nn_linear, True, None)


def test_embedding():
    data = torch.rand(2, 4, device='meta')

    # test layernorm
    ln = torch.nn.LayerNorm(4)
    _assert_output_shape(data, ln, patched_module.torch_nn_normalize, False, data.shape)

    # test group norm
    gn = torch.nn.GroupNorm(4, num_channels=2)
    _assert_output_shape(data, gn, patched_module.torch_nn_normalize, False, data.shape)

    # test batch norm 1d
    bn1d = torch.nn.BatchNorm1d(4)
    data = torch.rand(2, 4, device='meta')
    _assert_output_shape(data=data,
                         module=bn1d,
                         patch_fn=patched_module.torch_nn_normalize,
                         expect_exception=False,
                         output_shape=data.shape)

    data = torch.rand(2, 4, device='meta')
    _assert_output_shape(data=data,
                         module=bn1d,
                         patch_fn=patched_module.torch_nn_normalize,
                         expect_exception=False,
                         output_shape=data.shape)

    data = torch.rand(2, 3, 4, device='meta')
    _assert_output_shape(data=data,
                         module=bn1d,
                         patch_fn=patched_module.torch_nn_normalize,
                         expect_exception=False,
                         output_shape=data.shape)

    data = torch.rand(1, 2, 3, 4, device='meta')
    _assert_output_shape(data=data,
                         module=bn1d,
                         patch_fn=patched_module.torch_nn_normalize,
                         expect_exception=True,
                         output_shape=None)

    # test batch norm 2d
    bn2d = torch.nn.BatchNorm2d(4)

    data = torch.rand(1, 2, 3, 4, device='meta')
    _assert_output_shape(data=data,
                         module=bn2d,
                         patch_fn=patched_module.torch_nn_normalize,
                         expect_exception=False,
                         output_shape=data.shape)

    data = torch.rand(2, 3, 4, device='meta')
    _assert_output_shape(data=data,
                         module=bn2d,
                         patch_fn=patched_module.torch_nn_normalize,
                         expect_exception=True,
                         output_shape=None)

    # # test batch size 3d
    bn3d = torch.nn.BatchNorm3d(4)

    data = torch.rand(1, 1, 2, 3, 4, device='meta')
    _assert_output_shape(data=data,
                         module=bn3d,
                         patch_fn=patched_module.torch_nn_normalize,
                         expect_exception=False,
                         output_shape=data.shape)

    data = torch.rand(1, 2, 3, 4, device='meta')
    _assert_output_shape(data=data,
                         module=bn3d,
                         patch_fn=patched_module.torch_nn_normalize,
                         expect_exception=True,
                         output_shape=None)


def test_conv1d():
    # test conv 1d
    data = torch.rand(2, 3, 4)

    conv1d = torch.nn.Conv1d(in_channels=3, out_channels=4, kernel_size=2)
    materialized_output = conv1d(data)
    meta_data = data.to('meta')
    _assert_output_shape(data=meta_data,
                         module=conv1d,
                         patch_fn=patched_module.torch_nn_conv1d,
                         expect_exception=False,
                         output_shape=materialized_output.shape)

    conv1d = torch.nn.Conv1d(in_channels=3, out_channels=4, kernel_size=2, padding=1)
    materialized_output = conv1d(data)
    meta_data = data.to('meta')
    _assert_output_shape(data=meta_data,
                         module=conv1d,
                         patch_fn=patched_module.torch_nn_conv1d,
                         expect_exception=False,
                         output_shape=materialized_output.shape)

    conv1d = torch.nn.Conv1d(in_channels=3,
                             out_channels=4,
                             kernel_size=2,
                             padding=1,
                             dilation=2,
                             padding_mode='reflect')
    materialized_output = conv1d(data)
    meta_data = data.to('meta')
    _assert_output_shape(data=meta_data,
                         module=conv1d,
                         patch_fn=patched_module.torch_nn_conv1d,
                         expect_exception=False,
                         output_shape=materialized_output.shape)


def test_conv2d():
    # test conv 1d
    data = torch.rand(2, 3, 4, 4)
    conv2d = torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=2)
    materialized_output = conv2d(data)
    _assert_output_shape(data=data,
                         module=conv2d,
                         patch_fn=patched_module.torch_nn_conv2d,
                         expect_exception=False,
                         output_shape=materialized_output.shape)

    conv2d = torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=2, padding=1)
    materialized_output = conv2d(data)
    _assert_output_shape(data=data,
                         module=conv2d,
                         patch_fn=patched_module.torch_nn_conv2d,
                         expect_exception=False,
                         output_shape=materialized_output.shape)

    conv2d = torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=2, padding=1, dilation=2)
    materialized_output = conv2d(data)
    _assert_output_shape(data=data,
                         module=conv2d,
                         patch_fn=patched_module.torch_nn_conv2d,
                         expect_exception=False,
                         output_shape=materialized_output.shape)

    conv2d = torch.nn.Conv2d(in_channels=3,
                             out_channels=4,
                             kernel_size=2,
                             padding=1,
                             dilation=2,
                             padding_mode='reflect')
    materialized_output = conv2d(data)
    _assert_output_shape(data=data,
                         module=conv2d,
                         patch_fn=patched_module.torch_nn_conv2d,
                         expect_exception=False,
                         output_shape=materialized_output.shape)


def test_conv3d():
    # test conv 1d
    data = torch.rand(2, 3, 4, 4, 4)
    conv3d = torch.nn.Conv3d(in_channels=3, out_channels=4, kernel_size=2)
    materialized_output = conv3d(data)
    _assert_output_shape(data=data,
                         module=conv3d,
                         patch_fn=patched_module.torch_nn_conv3d,
                         expect_exception=False,
                         output_shape=materialized_output.shape)

    conv3d = torch.nn.Conv3d(in_channels=3, out_channels=4, kernel_size=2, padding=1)
    materialized_output = conv3d(data)
    _assert_output_shape(data=data,
                         module=conv3d,
                         patch_fn=patched_module.torch_nn_conv3d,
                         expect_exception=False,
                         output_shape=materialized_output.shape)

    conv3d = torch.nn.Conv3d(in_channels=3, out_channels=4, kernel_size=2, padding=1, dilation=2)
    materialized_output = conv3d(data)
    _assert_output_shape(data=data,
                         module=conv3d,
                         patch_fn=patched_module.torch_nn_conv3d,
                         expect_exception=False,
                         output_shape=materialized_output.shape)

    conv3d = torch.nn.Conv3d(in_channels=3,
                             out_channels=4,
                             kernel_size=2,
                             padding=1,
                             dilation=2,
                             padding_mode='reflect')
    materialized_output = conv3d(data)
    _assert_output_shape(data=data,
                         module=conv3d,
                         patch_fn=patched_module.torch_nn_conv3d,
                         expect_exception=False,
                         output_shape=materialized_output.shape)
