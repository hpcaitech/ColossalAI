import pytest
import torch

from colossalai.elixir.tracer.memory_tracer import MTensor
from colossalai.elixir.tracer.memory_tracer.op_cache import addmm_cache, bmm_cache, mm_cache
from colossalai.elixir.tracer.utils import get_cuda_allocated, get_cuda_max_allocated


def op_mm(x, y):
    u = torch.matmul(x, y)
    return u.shape


def op_addmm(x, y, z):
    u = torch.addmm(x, y, z)
    return u.shape


def op_bmm(x, y):
    u = torch.bmm(x, y)
    return u.shape


@pytest.mark.parametrize('dtype', [torch.float, torch.float16, torch.bfloat16])
def test_mm(dtype, size0=(4, 256), size1=(256, 1024)):
    torch.cuda.reset_peak_memory_stats()
    assert get_cuda_allocated() == 0

    x = torch.randn(size0, dtype=dtype, device='cuda')
    y = torch.randn(size1, dtype=dtype, device='cuda')
    torch_pre_alc = get_cuda_allocated()

    torch_z_size = op_mm(x, y)
    torch_temp_alc = get_cuda_max_allocated() - torch_pre_alc

    del x
    del y

    assert get_cuda_allocated() == 0
    x = MTensor(torch.randn(size0, dtype=dtype, device='cuda'))
    y = MTensor(torch.randn(size1, dtype=dtype, device='cuda'))
    op1_pre_alc = get_cuda_allocated()

    MTensor.reset_peak_memory()
    op1_z_size = op_mm(x, y)
    op1_temp_alc = MTensor.current_peak_memory() - op1_pre_alc

    assert torch_z_size == op1_z_size
    assert torch_pre_alc == op1_pre_alc
    assert torch_temp_alc == op1_temp_alc
    assert len(mm_cache.temp_memory) > 0

    MTensor.reset_peak_memory()
    op2_z_size = op_mm(x, y)
    op2_temp_alc = MTensor.current_peak_memory() - op1_pre_alc

    assert torch_z_size == op2_z_size
    assert torch_temp_alc == op2_temp_alc


@pytest.mark.parametrize('dtype', [torch.float, torch.float16, torch.bfloat16])
def test_addmm(dtype, size0=(4, 16), size1=(16, 64)):
    torch.cuda.reset_peak_memory_stats()
    assert get_cuda_allocated() == 0

    x = torch.randn(size0, dtype=dtype, device='cuda')
    y = torch.randn(size1, dtype=dtype, device='cuda')
    u = torch.randn(size1[-1], dtype=dtype, device='cuda')
    torch_pre_alc = get_cuda_allocated()

    torch_z_size = op_addmm(u, x, y)
    torch_temp_alc = get_cuda_max_allocated() - torch_pre_alc

    del x
    del y
    del u

    assert get_cuda_allocated() == 0
    x = MTensor(torch.randn(size0, dtype=dtype, device='cuda'))
    y = MTensor(torch.randn(size1, dtype=dtype, device='cuda'))
    u = MTensor(torch.randn(size1[-1], dtype=dtype, device='cuda'))
    op1_pre_alc = get_cuda_allocated()

    MTensor.reset_peak_memory()
    op1_z_size = op_addmm(u, x, y)
    op1_temp_alc = MTensor.current_peak_memory() - op1_pre_alc

    assert torch_z_size == op1_z_size
    assert torch_pre_alc == op1_pre_alc
    assert torch_temp_alc == op1_temp_alc
    assert len(addmm_cache.temp_memory) > 0

    MTensor.reset_peak_memory()
    op2_z_size = op_addmm(u, x, y)
    op2_temp_alc = MTensor.current_peak_memory() - op1_pre_alc

    assert torch_z_size == op2_z_size
    assert torch_temp_alc == op2_temp_alc


@pytest.mark.parametrize('dtype', [torch.float, torch.float16, torch.bfloat16])
def test_bmm(dtype, size0=(10, 4, 15), size1=(10, 15, 64)):
    torch.cuda.reset_peak_memory_stats()
    assert get_cuda_allocated() == 0

    x = torch.randn(size0, dtype=dtype, device='cuda')
    y = torch.randn(size1, dtype=dtype, device='cuda')
    torch_pre_alc = get_cuda_allocated()

    torch_z_size = op_bmm(x, y)
    torch_temp_alc = get_cuda_max_allocated() - torch_pre_alc

    del x
    del y

    assert get_cuda_allocated() == 0
    x = MTensor(torch.randn(size0, dtype=dtype, device='cuda'))
    y = MTensor(torch.randn(size1, dtype=dtype, device='cuda'))
    op1_pre_alc = get_cuda_allocated()

    MTensor.reset_peak_memory()
    op1_z_size = op_bmm(x, y)
    op1_temp_alc = MTensor.current_peak_memory() - op1_pre_alc

    assert torch_z_size == op1_z_size
    assert torch_pre_alc == op1_pre_alc
    assert torch_temp_alc == op1_temp_alc
    assert len(bmm_cache.temp_memory) > 0

    bmm_cache.print()

    MTensor.reset_peak_memory()
    op2_z_size = op_bmm(x, y)
    op2_temp_alc = MTensor.current_peak_memory() - op1_pre_alc

    assert torch_z_size == op2_z_size
    assert torch_temp_alc == op2_temp_alc


if __name__ == '__main__':
    test_addmm(dtype=torch.float)
