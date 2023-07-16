import torch
from torch import nn 
import torch.nn.functional as F

from colossalai.kernel.triton.ops import self_attention_compute_using_triton

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

    q = torch.transpose(q, 1, 2)
    k = torch.transpose(k, 1, 2)
    v = torch.transpose(v, 1, 2)

    k = torch.transpose(k, -1, -2)

    score_output = torch.einsum('bnij,bnjk->bnik', q, k)
    score_output *= scale

    score_output = F.softmax(score_output, dim = -1)
    res = torch.einsum('bnij,bnjk->bnik', score_output, v)
    res = torch.transpose(res, 1, 3)
    res = res.contiguous()


    return res.view(batches, -1, d_model)


          


def test_self_atttention_test():
    qkv = torch.randn((4, 24, 64*3), device="cuda", dtype=torch.float32)
    data_output_torch = self_attention_compute_using_torch(
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
    print(data_output_torch.shape)
    print(data_output_triton.shape)
    print(data_output_torch)
    print(data_output_triton)
    exit(0)
    print(torch.allclose(data_output_torch.cpu(), data_output_triton.cpu(), rtol=1e-3, atol=1e-3))




if __name__ == "__main__":
    test_self_atttention_test()