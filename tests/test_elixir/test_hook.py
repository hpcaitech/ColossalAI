from copy import deepcopy

import torch
import torch.nn as nn

from colossalai.elixir.hook import BufferStore, HookParam
from colossalai.elixir.tensor import FakeTensor


def test_hook():
    x = nn.Parameter(torch.randn(4, 4))

    ori_numel = x.numel()
    ori_size = x.size()
    ori_stride = x.stride()
    ori_offset = x.storage_offset()

    fake_data = FakeTensor(x.data)
    x.data = fake_data
    x.__class__ = HookParam

    assert x.numel() == ori_numel
    assert x.size() == ori_size
    assert x.stride() == ori_stride
    assert x.storage_offset() == ori_offset


def test_store():
    buffer = BufferStore(1024, torch.float16)
    print(buffer)

    x = torch.randn(4, 128, dtype=torch.float16, device='cuda')
    original_ptr_x = x.data_ptr()
    copy_x = deepcopy(x)

    y = torch.randn(512, dtype=torch.float16, device='cuda')
    original_ptr_y = y.data_ptr()
    copy_y = deepcopy(y)

    offset = 0
    offset = buffer.insert(x, offset)
    assert offset == x.numel()
    assert torch.equal(x, copy_x)

    offset = buffer.insert(y, offset)
    assert offset == 1024
    assert torch.equal(y, copy_y)

    buffer.erase(x)
    buffer.erase(y)
    assert x.data_ptr() == original_ptr_x
    assert y.data_ptr() == original_ptr_y


if __name__ == '__main__':
    test_store()
