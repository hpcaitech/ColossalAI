from abc import ABC
from collections import defaultdict
from typing import Iterable, NamedTuple

import torch
from torch.autograd.profiler_util import _format_memory


class BlockRequire(NamedTuple):
    numel: int
    dtype: torch.dtype


class TensorBlock(ABC):
    """TensorBlock is the memory unit of memory pool.
    It is a continuous memory block used to store tensors.
    Each chunk needs a corresponding TensorBlock to store its data during training.

    args:
        numel: the number of elements in the block
        dtype: the data type of the block
        device_type: the device type of the block
    """
    total_count: int = 0

    def __init__(self, numel: int, dtype: torch.dtype, device_type: str) -> None:
        self.block_id = TensorBlock.total_count
        TensorBlock.total_count += 1

        self.device_type = device_type
        self.payload: torch.Tensor = torch.empty((numel,), dtype=dtype, device=device_type)
        self.memo_occ: int = self.payload.numel() * self.payload.element_size()

    @property
    def numel(self):
        return self.payload.numel()

    @property
    def dtype(self):
        return self.payload.dtype

    @property
    def device(self):
        return self.payload.device

    def __hash__(self) -> int:
        return self.block_id

    def __eq__(self, other: object) -> bool:
        return self.block_id == other.block_id

    def __repr__(self) -> str:
        return f'(id={self.block_id}, numel={self.numel}, device={self.device_type}, dtype={self.dtype}, memo={self.memo_occ})'


class PublicBlock(TensorBlock):
    """Public blocks have the same length.
    Chunks of the same length can share the same public block.
    """

    def __init__(self, numel: int, dtype: torch.dtype, device_type: str) -> None:
        super().__init__(numel, dtype, device_type)
        self.block_type = 'public'

    def __repr__(self) -> str:
        return f'PublicBlock{super().__repr__()}'


class PrivateBlock(TensorBlock):
    """Private blocks may have different lengths.
    Each private chunk should use its own private block.
    """

    def __init__(self, numel: int, dtype: torch.dtype, device_type: str) -> None:
        super().__init__(numel, dtype, device_type)
        self.block_type = 'private'

    def __repr__(self) -> str:
        return f'PrivateBlock{super().__repr__()}'


class MemoryPool(object):
    """A memory pool consists of public blocks and private blocks.
    rCache uses memory pool to manage memory bolcks.
    Users should allocate memory blocks before using it.

    args:
        device_type: the device type of the memory pool
    """

    def __init__(self, device_type: str) -> None:
        self.device_type: str = device_type

        self.public_space: int = 0
        self.public_block_size: int = 0
        self.public_dtype: torch.dtype = None

        self.public_free_blocks: list = None
        self.public_used_blocks: set = None

        self.public_free_cnt: int = 0
        self.public_used_cnt: int = 0

        self.private_space: int = 0
        self.private_blocks: list = None
        self.private_lookup_dict: dict[BlockRequire, list] = None

        self.__allocate_flag = False

    def allocate(self,
                 public_dtype: torch.dtype = torch.float,
                 public_block_size: int = 1024,
                 public_block_number: int = 0,
                 private_block_list: Iterable[BlockRequire] = ()):
        assert self.__allocate_flag is False
        assert public_block_number >= 0

        self.public_free_blocks = list()
        self.public_used_blocks = set()
        for _ in range(public_block_number):
            block = PublicBlock(public_block_size, public_dtype, self.device_type)
            self.public_free_blocks.append(block)

        if public_block_number <= 0:
            self.public_space = 0
        else:
            self.public_space = self.public_free_blocks[0].memo_occ * public_block_number
        self.public_block_size = public_block_size
        self.public_dtype = public_dtype

        self.public_free_cnt = public_block_number
        self.public_used_cnt = 0

        self.private_space = 0
        self.private_blocks = list()
        self.private_lookup_dict = defaultdict(list)

        for require in private_block_list:
            block = PrivateBlock(require.numel, require.dtype, self.device_type)
            self.private_space += block.memo_occ
            self.private_blocks.append(block)
            self.private_lookup_dict[require].append(block)

        self.__allocate_flag = True

    def __repr__(self) -> str:
        return f'MP(public_space={_format_memory(self.public_space)}, private_space={_format_memory(self.private_space)})'

    def get_private_block(self, numel: int, dtype: torch.dtype):
        block_list = self.private_lookup_dict.get(BlockRequire(numel=numel, dtype=dtype))
        return block_list.pop()

    def get_public_block(self):
        self.public_free_cnt -= 1
        self.public_used_cnt += 1

        block = self.public_free_blocks.pop()
        self.public_used_blocks.add(block)

        return block

    def free_public_block(self, block: TensorBlock):
        assert isinstance(block, PublicBlock)
        assert block in self.public_used_blocks

        self.public_free_cnt += 1
        self.public_used_cnt -= 1

        self.public_used_blocks.remove(block)
        self.public_free_blocks.append(block)

        return block
