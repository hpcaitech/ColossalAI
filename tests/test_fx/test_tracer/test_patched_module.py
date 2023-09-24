import torch

from colossalai.fx.tracer.meta_patch import patched_module
from colossalai.testing import clear_cache_before_run


def _run(data, module, patch_fn):
    try:
        if isinstance(data, dict):
            output = patch_fn(module, **data)
        if isinstance(data, tuple) or isinstance(data, list):
            output = patch_fn(module, *data)
        else:
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
        if isinstance(output, tuple):
            for item, shape in zip(output, output_shape):
                assert item.is_meta
                assert item.shape == shape
        else:
            assert output.is_meta
            assert output.shape == output_shape


@clear_cache_before_run()
def test_linear():
    # test linear patch can produce the meta output with correct shape
    data = torch.rand(2, 4, device="meta")
    module = torch.nn.Linear(4, 2)
    _assert_output_shape(data, module, patched_module.torch_nn_linear, False, torch.Size([2, 2]))

    # test if the linear patch can catch exception when dimension does not match
    data = torch.rand(2, 2, device="meta")
    _assert_output_shape(data, module, patched_module.torch_nn_linear, True, None)


@clear_cache_before_run()
def test_rnn():
    # test rnn patch can produce the meta output with correct shape
    data = (torch.randn(5, 3, 10), torch.randn(2, 3, 20))
    module = torch.nn.RNN(10, 20, 2)
    output, hn = module(*data)
    meta_data = (torch.randn(5, 3, 10).to("meta"), torch.randn(2, 3, 20).to("meta"))
    _assert_output_shape(meta_data, module, patched_module.torch_nn_rnn, False, (output.shape, hn.shape))

    # test if the rnn patch can catch exception when dimension does not match
    data = (torch.randn(5, 3, 10), torch.randn(2, 3, 20))
    module = torch.nn.RNN(10, 20, 2)
    output, hn = module(*data)
    meta_data = (torch.randn(5, 3, 1).to("meta"), torch.randn(2, 3, 20).to("meta"))
    _assert_output_shape(meta_data, module, patched_module.torch_nn_rnn, True, None)


@clear_cache_before_run()
def test_embedding():
    data = torch.rand(2, 4, device="meta")

    # test layernorm
    ln = torch.nn.LayerNorm(4)
    _assert_output_shape(data, ln, patched_module.torch_nn_normalize, False, data.shape)

    # test group norm
    gn = torch.nn.GroupNorm(4, num_channels=8)
    _assert_output_shape(data, gn, patched_module.torch_nn_normalize, False, data.shape)

    # test batch norm 1d
    bn1d = torch.nn.BatchNorm1d(4)
    data = torch.rand(2, 4, device="meta")
    _assert_output_shape(
        data=data,
        module=bn1d,
        patch_fn=patched_module.torch_nn_normalize,
        expect_exception=False,
        output_shape=data.shape,
    )

    data = torch.rand(2, 4, device="meta")
    _assert_output_shape(
        data=data,
        module=bn1d,
        patch_fn=patched_module.torch_nn_normalize,
        expect_exception=False,
        output_shape=data.shape,
    )

    data = torch.rand(2, 3, 4, device="meta")
    _assert_output_shape(
        data=data,
        module=bn1d,
        patch_fn=patched_module.torch_nn_normalize,
        expect_exception=False,
        output_shape=data.shape,
    )

    data = torch.rand(1, 2, 3, 4, device="meta")
    _assert_output_shape(
        data=data, module=bn1d, patch_fn=patched_module.torch_nn_normalize, expect_exception=True, output_shape=None
    )

    # test batch norm 2d
    bn2d = torch.nn.BatchNorm2d(4)

    data = torch.rand(1, 2, 3, 4, device="meta")
    _assert_output_shape(
        data=data,
        module=bn2d,
        patch_fn=patched_module.torch_nn_normalize,
        expect_exception=False,
        output_shape=data.shape,
    )

    data = torch.rand(2, 3, 4, device="meta")
    _assert_output_shape(
        data=data, module=bn2d, patch_fn=patched_module.torch_nn_normalize, expect_exception=True, output_shape=None
    )

    # # test batch size 3d
    bn3d = torch.nn.BatchNorm3d(4)

    data = torch.rand(1, 1, 2, 3, 4, device="meta")
    _assert_output_shape(
        data=data,
        module=bn3d,
        patch_fn=patched_module.torch_nn_normalize,
        expect_exception=False,
        output_shape=data.shape,
    )

    data = torch.rand(1, 2, 3, 4, device="meta")
    _assert_output_shape(
        data=data, module=bn3d, patch_fn=patched_module.torch_nn_normalize, expect_exception=True, output_shape=None
    )


@clear_cache_before_run()
def test_conv1d():
    # test conv 1d
    data = torch.rand(2, 3, 4)

    conv1d = torch.nn.Conv1d(in_channels=3, out_channels=4, kernel_size=2)
    materialized_output = conv1d(data)
    meta_data = data.to("meta")
    _assert_output_shape(
        data=meta_data,
        module=conv1d,
        patch_fn=patched_module.torch_nn_conv1d,
        expect_exception=False,
        output_shape=materialized_output.shape,
    )

    conv1d = torch.nn.Conv1d(in_channels=3, out_channels=4, kernel_size=2, padding=1)
    materialized_output = conv1d(data)
    meta_data = data.to("meta")
    _assert_output_shape(
        data=meta_data,
        module=conv1d,
        patch_fn=patched_module.torch_nn_conv1d,
        expect_exception=False,
        output_shape=materialized_output.shape,
    )

    conv1d = torch.nn.Conv1d(
        in_channels=3, out_channels=4, kernel_size=2, padding=1, dilation=2, padding_mode="reflect"
    )
    materialized_output = conv1d(data)
    meta_data = data.to("meta")
    _assert_output_shape(
        data=meta_data,
        module=conv1d,
        patch_fn=patched_module.torch_nn_conv1d,
        expect_exception=False,
        output_shape=materialized_output.shape,
    )


def test_conv2d():
    # test conv 2d
    data = torch.rand(2, 3, 4, 4)
    conv2d = torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=2)
    materialized_output = conv2d(data)
    _assert_output_shape(
        data=data,
        module=conv2d,
        patch_fn=patched_module.torch_nn_conv2d,
        expect_exception=False,
        output_shape=materialized_output.shape,
    )

    conv2d = torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=2, padding=1)
    materialized_output = conv2d(data)
    _assert_output_shape(
        data=data,
        module=conv2d,
        patch_fn=patched_module.torch_nn_conv2d,
        expect_exception=False,
        output_shape=materialized_output.shape,
    )

    conv2d = torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=2, padding=1, dilation=2)
    materialized_output = conv2d(data)
    _assert_output_shape(
        data=data,
        module=conv2d,
        patch_fn=patched_module.torch_nn_conv2d,
        expect_exception=False,
        output_shape=materialized_output.shape,
    )

    conv2d = torch.nn.Conv2d(
        in_channels=3, out_channels=4, kernel_size=2, padding=1, dilation=2, padding_mode="reflect"
    )
    materialized_output = conv2d(data)
    _assert_output_shape(
        data=data,
        module=conv2d,
        patch_fn=patched_module.torch_nn_conv2d,
        expect_exception=False,
        output_shape=materialized_output.shape,
    )


@clear_cache_before_run()
def test_conv3d():
    # test conv 3d
    data = torch.rand(2, 3, 4, 4, 4)
    conv3d = torch.nn.Conv3d(in_channels=3, out_channels=4, kernel_size=2)
    materialized_output = conv3d(data)
    _assert_output_shape(
        data=data,
        module=conv3d,
        patch_fn=patched_module.torch_nn_conv3d,
        expect_exception=False,
        output_shape=materialized_output.shape,
    )

    conv3d = torch.nn.Conv3d(in_channels=3, out_channels=4, kernel_size=2, padding=1)
    materialized_output = conv3d(data)
    _assert_output_shape(
        data=data,
        module=conv3d,
        patch_fn=patched_module.torch_nn_conv3d,
        expect_exception=False,
        output_shape=materialized_output.shape,
    )

    conv3d = torch.nn.Conv3d(in_channels=3, out_channels=4, kernel_size=2, padding=1, dilation=2)
    materialized_output = conv3d(data)
    _assert_output_shape(
        data=data,
        module=conv3d,
        patch_fn=patched_module.torch_nn_conv3d,
        expect_exception=False,
        output_shape=materialized_output.shape,
    )

    conv3d = torch.nn.Conv3d(
        in_channels=3, out_channels=4, kernel_size=2, padding=1, dilation=2, padding_mode="reflect"
    )
    materialized_output = conv3d(data)
    _assert_output_shape(
        data=data,
        module=conv3d,
        patch_fn=patched_module.torch_nn_conv3d,
        expect_exception=False,
        output_shape=materialized_output.shape,
    )


@clear_cache_before_run()
def test_conv_transpose1d():
    # test conv transpose1d
    data = torch.rand(2, 3, 4)

    convtrans1d = torch.nn.ConvTranspose1d(in_channels=3, out_channels=4, kernel_size=2)
    materialized_output = convtrans1d(data)
    meta_data = data.to("meta")
    _assert_output_shape(
        data=meta_data,
        module=convtrans1d,
        patch_fn=patched_module.torch_nn_convtranspose1d,
        expect_exception=False,
        output_shape=materialized_output.shape,
    )

    convtrans1d = torch.nn.ConvTranspose1d(in_channels=3, out_channels=4, kernel_size=2, padding=1)
    materialized_output = convtrans1d(data)
    meta_data = data.to("meta")
    _assert_output_shape(
        data=meta_data,
        module=convtrans1d,
        patch_fn=patched_module.torch_nn_convtranspose1d,
        expect_exception=False,
        output_shape=materialized_output.shape,
    )


@clear_cache_before_run()
def test_conv_transpose2d():
    # test conv transpose2d
    data = torch.rand(2, 3, 4, 4)

    convtrans2d = torch.nn.ConvTranspose2d(in_channels=3, out_channels=4, kernel_size=2)
    materialized_output = convtrans2d(data)
    meta_data = data.to("meta")
    _assert_output_shape(
        data=meta_data,
        module=convtrans2d,
        patch_fn=patched_module.torch_nn_convtranspose2d,
        expect_exception=False,
        output_shape=materialized_output.shape,
    )

    convtrans2d = torch.nn.ConvTranspose2d(in_channels=3, out_channels=4, kernel_size=2, padding=1)
    materialized_output = convtrans2d(data)
    meta_data = data.to("meta")
    _assert_output_shape(
        data=meta_data,
        module=convtrans2d,
        patch_fn=patched_module.torch_nn_convtranspose2d,
        expect_exception=False,
        output_shape=materialized_output.shape,
    )


@clear_cache_before_run()
def test_conv_transpose3d():
    # test conv transpose2d
    data = torch.rand(2, 3, 4, 4, 4)

    convtrans3d = torch.nn.ConvTranspose3d(in_channels=3, out_channels=4, kernel_size=2)
    materialized_output = convtrans3d(data)
    meta_data = data.to("meta")
    _assert_output_shape(
        data=meta_data,
        module=convtrans3d,
        patch_fn=patched_module.torch_nn_convtranspose3d,
        expect_exception=False,
        output_shape=materialized_output.shape,
    )

    convtrans3d = torch.nn.ConvTranspose3d(in_channels=3, out_channels=4, kernel_size=2, padding=1)
    materialized_output = convtrans3d(data)
    meta_data = data.to("meta")
    _assert_output_shape(
        data=meta_data,
        module=convtrans3d,
        patch_fn=patched_module.torch_nn_convtranspose3d,
        expect_exception=False,
        output_shape=materialized_output.shape,
    )


@clear_cache_before_run()
def test_pool1d():
    combinations = [
        [torch.nn.MaxPool1d, patched_module.torch_nn_maxpool1d],
        [torch.nn.AvgPool1d, patched_module.torch_nn_avgpool1d],
    ]

    for layer_cls, patch_func in combinations:
        pooler = layer_cls(kernel_size=3)

        data = torch.rand(2, 3, 4)
        materialized_output = pooler(data)
        _assert_output_shape(
            data=data,
            module=pooler,
            patch_fn=patch_func,
            expect_exception=False,
            output_shape=materialized_output.shape,
        )

        data = torch.rand(2, 4)
        materialized_output = pooler(data)
        _assert_output_shape(
            data=data,
            module=pooler,
            patch_fn=patch_func,
            expect_exception=False,
            output_shape=materialized_output.shape,
        )

        data = torch.rand(2, 3, 4, 4)
        _assert_output_shape(data=data, module=pooler, patch_fn=patch_func, expect_exception=True, output_shape=None)


@clear_cache_before_run()
def test_pool2d():
    combinations = [
        [torch.nn.MaxPool2d, patched_module.torch_nn_maxpool2d],
        [torch.nn.AvgPool2d, patched_module.torch_nn_avgpool2d],
    ]

    for layer_cls, patch_func in combinations:
        pooler = layer_cls(kernel_size=3)

        # test max pool 3d
        data = torch.rand(2, 3, 4, 4)
        materialized_output = pooler(data)
        _assert_output_shape(
            data=data,
            module=pooler,
            patch_fn=patch_func,
            expect_exception=False,
            output_shape=materialized_output.shape,
        )

        # test max pool 3d
        data = torch.rand(2, 4, 4)
        materialized_output = pooler(data)
        _assert_output_shape(
            data=data,
            module=pooler,
            patch_fn=patch_func,
            expect_exception=False,
            output_shape=materialized_output.shape,
        )

        # test max pool 3d
        data = torch.rand(2, 3, 4, 4, 4)
        _assert_output_shape(data=data, module=pooler, patch_fn=patch_func, expect_exception=True, output_shape=None)


@clear_cache_before_run()
def test_pool3d():
    combinations = [
        [torch.nn.MaxPool3d, patched_module.torch_nn_maxpool3d],
        [torch.nn.AvgPool3d, patched_module.torch_nn_avgpool3d],
    ]

    for layer_cls, patch_func in combinations:
        pooler = layer_cls(kernel_size=3)

        # test max pool 3d
        data = torch.rand(2, 3, 4, 4, 4)
        materialized_output = pooler(data)
        _assert_output_shape(
            data=data,
            module=pooler,
            patch_fn=patch_func,
            expect_exception=False,
            output_shape=materialized_output.shape,
        )

        # test max pool 3d
        data = torch.rand(2, 4, 4, 4)
        materialized_output = pooler(data)
        _assert_output_shape(
            data=data,
            module=pooler,
            patch_fn=patch_func,
            expect_exception=False,
            output_shape=materialized_output.shape,
        )

        # test max pool 3d
        data = torch.rand(2, 3, 4)
        _assert_output_shape(data=data, module=pooler, patch_fn=patch_func, expect_exception=True, output_shape=None)


# adapative pooling is different from other pooling, so test it individually
@clear_cache_before_run()
def test_adaptive_pooling_1d():
    pooler = torch.nn.AdaptiveAvgPool1d(output_size=3)
    patch_func = patched_module.torch_nn_adapative_pooling_1d

    data = torch.rand(3, 4)
    output = pooler(data)
    _assert_output_shape(
        data=data, module=pooler, patch_fn=patch_func, expect_exception=False, output_shape=output.shape
    )

    data = torch.rand(2, 3, 4)
    output = pooler(data)
    _assert_output_shape(
        data=data, module=pooler, patch_fn=patch_func, expect_exception=False, output_shape=output.shape
    )

    data = torch.rand(2, 3, 4, 5)
    _assert_output_shape(data=data, module=pooler, patch_fn=patch_func, expect_exception=True, output_shape=None)


@clear_cache_before_run()
def test_adaptive_pooling_2d():
    pooler = torch.nn.AdaptiveAvgPool2d(output_size=3)
    patch_func = patched_module.torch_nn_adapative_pooling_2d

    data = torch.rand(3, 4)
    _assert_output_shape(data=data, module=pooler, patch_fn=patch_func, expect_exception=True, output_shape=None)

    data = torch.rand(2, 3, 4)
    output = pooler(data)
    _assert_output_shape(
        data=data, module=pooler, patch_fn=patch_func, expect_exception=False, output_shape=output.shape
    )

    data = torch.rand(2, 3, 4, 5)
    output = pooler(data)
    _assert_output_shape(
        data=data, module=pooler, patch_fn=patch_func, expect_exception=False, output_shape=output.shape
    )


@clear_cache_before_run()
def test_adaptive_pooling_3d():
    pooler = torch.nn.AdaptiveAvgPool3d(output_size=3)
    patch_func = patched_module.torch_nn_adapative_pooling_3d

    data = torch.rand(3, 4, 5)
    _assert_output_shape(data=data, module=pooler, patch_fn=patch_func, expect_exception=True, output_shape=None)

    data = torch.rand(2, 3, 4, 5)
    output = pooler(data)
    _assert_output_shape(
        data=data, module=pooler, patch_fn=patch_func, expect_exception=False, output_shape=output.shape
    )

    data = torch.rand(2, 3, 4, 5, 6)
    output = pooler(data)
    _assert_output_shape(
        data=data, module=pooler, patch_fn=patch_func, expect_exception=False, output_shape=output.shape
    )
