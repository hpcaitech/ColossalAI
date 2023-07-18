import pytest
from packaging import version
import torch
from torch import nn 
import torch.nn.functional as F

from colossalai.kernel.triton.ops import self_attention_compute_using_triton
from colossalai.kernel.triton.qkv_matmul_kernel import qkv_gemm_4d_kernel

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("please install triton from https://github.com/openai/triton")

TRITON_CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse('11.4')

@pytest.mark.skipif(not TRITON_CUDA_SUPPORT, reason="triton requires cuda version to be higher than 11.4")
def test_qkv_matmul():
    qkv = torch.randn((4, 24, 64*3), device="cuda", dtype=torch.float16)
    scale = 1.2
    head_size = 32
    batches = qkv.shape[0]
    d_model = qkv.shape[-1] // 3
    num_of_heads = d_model // head_size

    q = qkv[:, :, :d_model]
    k = qkv[:, :, d_model:d_model * 2]

    q = q.view(batches, -1, num_of_heads, head_size)
    k = k.view(batches, -1, num_of_heads, head_size)
    q_copy = q.clone()
    k_copy = k.clone()
    q = torch.transpose(q, 1, 2).contiguous()
    k = torch.transpose(k, 1, 2).contiguous()
    k = torch.transpose(k, 2, 3).contiguous()

    torch_ouput = torch.einsum('bnij,bnjk->bnik', q, k)
    torch_ouput *= 1.2

    q, k = q_copy, k_copy
    batches, M, H, K = q.shape
    N = k.shape[1]
    score_output = torch.empty(
    (batches, H, M, N), device=q.device, dtype=q.dtype)

    grid = lambda meta: (
        batches,
        H,
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) *
        triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )

    K = q.shape[3]
    qkv_gemm_4d_kernel[grid](
        q, k, score_output,
        M, N, K,
        q.stride(0), q.stride(2), q.stride(1), q.stride(3),
        k.stride(0), k.stride(2), k.stride(3), k.stride(1),
        score_output.stride(0), score_output.stride(1), score_output.stride(2), score_output.stride(3),
        scale=scale,
        # currently manually setting, later on we can use auto-tune config to match best setting
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
    )

    check = torch.allclose(torch_ouput.cpu(), score_output.cpu(), rtol=1e-3, atol=1e-5)
    assert check is True, "the outputs of triton and torch are not matched"
    

def self_attention_compute_using_torch(qkv,
                                       input_mask,
                                       scale,
                                       head_size
                                       ):

    batches = qkv.shape[0]
    d_model = qkv.shape[-1] // 3
    num_of_heads = d_model // head_size
    
    q = qkv[:, :, :d_model]
    k = qkv[:, :, d_model:d_model * 2]
    v = qkv[:, :, d_model * 2:]
    q = q.view(batches, -1, num_of_heads, head_size)
    k = k.view(batches, -1, num_of_heads, head_size)
    v = v.view(batches, -1, num_of_heads, head_size)

    q = torch.transpose(q, 1, 2).contiguous()
    k = torch.transpose(k, 1, 2).contiguous()
    v = torch.transpose(v, 1, 2).contiguous()

    k = torch.transpose(k, -1, -2).contiguous()

    score_output = torch.einsum('bnij,bnjk->bnik', q, k)
    score_output *= scale

    softmax_output = F.softmax(score_output, dim = -1)
    res = torch.einsum('bnij,bnjk->bnik', softmax_output, v)
    res = torch.transpose(res, 1, 2)
    res = res.contiguous()


    return res.view(batches, -1, d_model), score_output, softmax_output

@pytest.mark.skipif(not TRITON_CUDA_SUPPORT, reason="triton requires cuda version to be higher than 11.4")
def test_self_atttention_test():

    qkv = torch.randn((4, 24, 64*3), device="cuda", dtype=torch.float16)
    data_output_torch, score_output_torch, softmax_output_torch = self_attention_compute_using_torch(
                                                           qkv.clone(), 
                                                           input_mask = None, 
                                                           scale = 1.2, 
                                                           head_size = 32
                                                           )

    data_output_triton = self_attention_compute_using_triton(
                                                            qkv.clone(),
                                                            alibi=None,
                                                            head_size=32,
                                                            scale=1.2,
                                                            input_mask=None,
                                                            layer_past=None,
                                                            use_flash=False,
                                                            triangular=True)

    check = torch.allclose(data_output_triton.cpu(), data_output_torch.cpu(), rtol=1e-4, atol=1e-2)
    assert check is True, "the triton output is not matched with torch output"


if __name__ == "__main__":
    test_qkv_matmul()
    test_self_atttention_test()