import torch

from colossalai.elixir.chunk import BlockRequire, MemoryPool, PrivateBlock, PublicBlock
from colossalai.testing import run_on_environment_flag


@run_on_environment_flag('ELX')
def test_block():
    b = PublicBlock(123, torch.float16, 'cuda')
    payload_b = b.payload

    assert payload_b.numel() == 123
    assert payload_b.dtype == torch.float16
    assert payload_b.device.type == 'cuda'
    assert payload_b.numel() * payload_b.element_size() == b.memo_occ

    c = PrivateBlock(77, torch.float, 'cpu')
    payload_c = c.payload

    assert payload_c.numel() == 77
    assert payload_c.dtype == torch.float
    assert payload_c.device.type == 'cpu'
    assert payload_c.numel() * payload_c.element_size() == c.memo_occ

    print('test_block: ok')


@run_on_environment_flag('ELX')
def test_memory_pool():
    mp = MemoryPool(device_type='cuda')
    private_list = [BlockRequire(5, torch.float), BlockRequire(81, torch.float16)]
    mp.allocate(public_block_number=4, private_block_list=private_list)

    block0 = mp.get_public_block()

    assert block0 in mp.public_used_blocks
    assert mp.public_used_cnt == 1
    assert mp.public_free_cnt == 3

    block1 = mp.get_public_block()

    assert block1 in mp.public_used_blocks
    assert mp.public_used_cnt == 2
    assert mp.public_free_cnt == 2

    mp.free_public_block(block0)
    mp.free_public_block(block1)

    assert block0 in mp.public_free_blocks
    assert block1 in mp.public_free_blocks
    assert mp.public_used_cnt == 0
    assert mp.public_free_cnt == 4

    block0 = mp.get_private_block(5, torch.float)
    assert block0.numel == 5
    assert block0.dtype == torch.float

    print('test_memory_pool: ok')


if __name__ == '__main__':
    test_block()
    test_memory_pool()
