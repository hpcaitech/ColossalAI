import torch

from colossalai.elixir.chunk import BlockSpec, MemoryPool, PrivateBlock, PublicBlock
from colossalai.testing import run_on_environment_flag


def test_block():
    # test for public block
    public_block = PublicBlock(123, torch.float16, 'cuda')
    public_payload = public_block.payload

    assert public_payload.numel() == 123
    assert public_payload.dtype == torch.float16
    assert public_payload.device.type == 'cuda'
    assert public_payload.numel() * public_payload.element_size() == public_block.size_in_bytes

    # test for private block
    private_block = PrivateBlock(77, torch.float, 'cpu')
    private_payload = private_block.payload

    assert private_payload.numel() == 77
    assert private_payload.dtype == torch.float
    assert private_payload.device.type == 'cpu'
    assert private_payload.numel() * private_payload.element_size() == private_block.size_in_bytes
    print('test_block: ok')


def test_memory_pool():
    mp = MemoryPool(device_type='cuda')

    # allocate public blocks
    mp.allocate_public_blocks(block_num=4)

    # allocate private blocks
    private_block_specs = [BlockSpec(5, torch.float), BlockSpec(81, torch.float16)]
    mp.allocate_private_blocks(private_block_specs)

    # test for public blocks
    block0 = mp.pop_public_block()
    assert block0 in mp.public_used_blocks
    assert mp.public_used_count == 1
    assert mp.public_free_count == 3

    block1 = mp.pop_public_block()
    assert block1 in mp.public_used_blocks
    assert mp.public_used_count == 2
    assert mp.public_free_count == 2

    mp.free_public_block(block0)
    mp.free_public_block(block1)
    assert block0 in mp.public_free_blocks
    assert block1 in mp.public_free_blocks
    assert mp.public_used_count == 0
    assert mp.public_free_count == 4

    # test for private block
    block0 = mp.get_private_block(5, torch.float)
    assert block0.numel == 5
    assert block0.dtype == torch.float
    print('test_memory_pool: ok')


if __name__ == '__main__':
    test_block()
    test_memory_pool()
