import torch
try:
    from col_linear_lib import dense_layer_fp32_forward, dense_layer_fp16_forward, batch_dense_layer_fp16_forward
    HAS_FLASH_CUDA = True
except:
    HAS_FLASH_CUDA = False
    print("in order to use flash-attention, make sure you install cuda kernels in op directory")


if HAS_FLASH_CUDA:
    def linear(data, weight):
        data_shape = None
        if len(data.shape) > 2:
            data_shape = data.shape
            data = data.view(-1, data.shape[-1])

        assert data.dtype == torch.float16, "only fp16 precision supports"
        assert len(data.shape) == 2, "the shape must be 2-D"
        assert len(weight.shape) == 2, "the shape must be 2-D"

        M, K = data.shape
        _, N = weight.shape

        assert K == weight.shape[0], "the shape is not matchted"

        out = torch.empty((M, N), device=data.get_device(), dtype=torch.float16)
        dense_layer_fp16_forward(data, weight, out, 99)
        if data_shape is not None:
            out = out.view(*data_shape[:-1], N)
        return out


    def batch_linear(data, weight, alibi = None, alpha = 1, beta = 0):
        """
        it is equivalent to alibi.bmm(data, weight)
        only supports float16
        """
        batch_count, M, K = data.shape
        _, N = weight.shape
        assert data.shape[-1] == weight.shape[0], "the k-dimensions must be matched"
        if alibi is None:
            out = torch.empty((batch_count, M, N), dtype=torch.float16, device=data.get_device())
        else:
            out = alibi

        batch_dense_layer_fp16_forward(data, weight, out, alpha, beta, 99)
        return out
    