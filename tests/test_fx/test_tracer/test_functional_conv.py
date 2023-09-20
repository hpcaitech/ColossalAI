import torch
from torch.nn import functional as F

from colossalai.fx.tracer.meta_patch import patched_function
from colossalai.testing import clear_cache_before_run


@clear_cache_before_run()
def test_conv():
    # test F.conv_1d
    data_1d = torch.rand(3, 16, 10)
    weight_1d = torch.rand(3, 16, 3)
    out_1d = F.conv1d(data_1d, weight_1d)
    patched_out_1d = patched_function.torch_nn_functional_conv1d(data_1d, weight_1d)
    assert out_1d.shape == patched_out_1d.shape

    # test F.conv_transpose1d
    weight_1d = torch.transpose(weight_1d, 0, 1)
    out_transpose_1d = F.conv_transpose1d(data_1d, weight_1d)
    patched_out_transpose_1d = patched_function.torch_nn_functional_convtranspose1d(data_1d, weight_1d)
    assert out_transpose_1d.shape == patched_out_transpose_1d.shape

    # test F.conv2d
    data_2d = torch.rand(3, 16, 10, 10)
    weight_2d = torch.rand(3, 16, 3, 3)
    out_2d = F.conv2d(data_2d, weight_2d)
    patched_out_2d = patched_function.torch_nn_functional_conv2d(data_2d, weight_2d)
    assert out_2d.shape == patched_out_2d.shape

    # test F.conv_transpose2d
    weight_2d = torch.transpose(weight_2d, 0, 1)
    out_transpose_2d = F.conv_transpose2d(data_2d, weight_2d)
    patched_out_transpose_2d = patched_function.torch_nn_functional_convtranspose2d(data_2d, weight_2d)
    assert out_transpose_2d.shape == patched_out_transpose_2d.shape

    # test F.conv3d
    data_3d = torch.rand(3, 16, 10, 10, 10)
    weight_3d = torch.rand(3, 16, 3, 3, 3)
    out_3d = F.conv3d(data_3d, weight_3d)
    patched_out_3d = patched_function.torch_nn_functional_conv3d(data_3d, weight_3d)
    assert out_3d.shape == patched_out_3d.shape

    # test F.conv_transpose3d
    weight_3d = torch.transpose(weight_3d, 0, 1)
    out_transpose_3d = F.conv_transpose3d(data_3d, weight_3d)
    patched_out_transpose_3d = patched_function.torch_nn_functional_convtranspose3d(data_3d, weight_3d)
    assert out_transpose_3d.shape == patched_out_transpose_3d.shape


if __name__ == "__main__":
    test_conv()
