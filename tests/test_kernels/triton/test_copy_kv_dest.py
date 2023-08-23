import pytest
from packaging import version

import torch
from torch import nn

from tests.test_kernels.triton.utils import benchmark
from colossalai.kernel.triton.copy_kv_cache_dest import copy_kv_cache_to_dest

TRITON_CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse('11.4')

@pytest.mark.skipif(not TRITON_CUDA_SUPPORT, reason="triton requires cuda version to be higher than 11.4")
def test_kv_cache_copy_op():
    
    B_NTX = 32 * 2048
    head_num = 8
    head_dim = 64
    
    cache = torch.randn((B_NTX, head_num, head_dim), device="cuda", dtype=torch.float16)
    dest_index = torch.arange(0, B_NTX, device="cuda", dtype=torch.int32)
    
    dest_data = torch.ones((B_NTX, head_num, head_dim), device="cuda", dtype=torch.float16)
    
    copy_kv_cache_to_dest(cache, dest_index, dest_data)
    
    assert torch.allclose(cache.cpu(), dest_data.cpu(), rtol=1e-3, atol=1e-3), "copy_kv_cache_to_dest outputs from triton and torch are not matched"
    
    latency = benchmark(copy_kv_cache_to_dest, cache, dest_index, dest_data)
    print("the average latency is {} ms".format(str(latency)))
    

if __name__ == "__main__":
    test_kv_cache_copy_op()

