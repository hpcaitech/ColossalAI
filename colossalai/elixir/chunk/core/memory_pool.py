from abc import ABC
from collections import defaultdict
from enum import Enum
from typing import Iterable, NamedTuple

import torch
from torch.autograd.profiler_util import _format_memory


class BlockSpec(NamedTuple):
    """
    BlockSpec is the specification of a block. It contains the number of elements and the data type of the block.

    Args:
        numel (int): the number of elements in the block
        dtype (torch.dtype): the data type of the block
    """
    numel: int
    dtype: torch.dtype


class BlockType(Enum):
    """
    BlockType is the type of a block. There are two types of blocks: public and private.
    """
    PUBLIC = 0
    PRIVATE = 1


class TensorBlock(ABC):
    """
    TensorBlock is the memory unit of memory pool. It is a contiguous memory block used to store tensors.
    Each chunk needs a corresponding TensorBlock to store its data during training.

    args:
        size (int): the number of elements in the block
        dtype (torch.dtype): the data type of the block
        device_type (str): the device type of the block
    """
    total_count: int = 0

    def __init__(self, size: int, dtype: torch.dtype, device_type: str, block_type: BlockType) -> None:
        self.block_id = TensorBlock.total_count
        TensorBlock.total_count += 1

        self.device_type = device_type
        self.payload: torch.Tensor = torch.empty((size,), dtype=dtype, device=device_type)
        self.size_in_bytes: int = self.payload.numel() * self.payload.element_size()
        self.block_type = block_type

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
        return f'{self.block_type}(\n\tID = {self.block_id}, \n\tsize = {self.numel}, \n\tdevice = {self.device_type}, \n\tdtype = {self.dtype}, \n\tsize in bytes={self.size_in_bytes}\n)'


class PublicBlock(TensorBlock):
    """
    Public blocks have the same length. Chunks of the same length can share the same public block.
    """

    def __init__(self, numel: int, dtype: torch.dtype, device_type: str) -> None:
        super().__init__(numel, dtype, device_type, BlockType.PUBLIC)


class PrivateBlock(TensorBlock):
    """
    Private blocks may have different lengths. Each private chunk should use its own private block.
    """

    def __init__(self, numel: int, dtype: torch.dtype, device_type: str) -> None:
        super().__init__(numel, dtype, device_type, BlockType.PRIVATE)


class MemoryPool(object):
    """
    A memory pool consists of public blocks and private blocks.
    rCache uses memory pool to manage memory bolcks.
    Users should allocate memory blocks before using it.

    args:
        device_type (str): the device type of the memory pool
    """

    def __init__(self, device_type: str) -> None:
        assert device_type in [
            'cuda', 'cpu'
        ], f'Expected device type to be cuda or cpu, but got an invalid device type: {device_type}'
        self.device_type: str = device_type

        # public space
        # public space = number of public block x the public block size in bytes
        # all public blocks have the same block size
        self.public_space: int = 0
        self.public_block_size: int = 0
        self.public_dtype: torch.dtype = None

        # create block holder and counter
        self.public_free_blocks: list = list()
        self.public_used_blocks: set = set()
        self.public_free_count: int = 0
        self.public_used_count: int = 0

        # private space
        # private look up dict returns an empty list if the block is not found
        self.private_space: int = 0
        self.private_blocks: list = list()
        self.private_lookup_dict: dict[BlockSpec, list] = defaultdict(list)

        # flags for block allcation
        self.__public_allocated_flag = False
        self.__private_allocated_flag = False

    def allocate_public_blocks(self, block_num: int, block_spec: BlockSpec = None):
        """
        Allocate public tensor blocks for the memory pool. This method will allocate public_block_number blocks with size equal to public_block_size.
        """
        assert not self.__public_allocated_flag, 'Public blocks have been allocated to this MemoryPool object, it is not allowed to allocate again.'
        assert block_num >= 0, f'Expected public_block_number >= 0, but got {block_num}'

        if block_spec is None:
            block_spec = BlockSpec(numel=1024, dtype=torch.float)

        # allocate public blocks
        for _ in range(block_num):
            block = PublicBlock(block_spec.numel, block_spec.dtype, self.device_type)
            self.public_free_blocks.append(block)
            self.public_space += block.size_in_bytes
            self.public_free_count += 1

        # store the block spec info
        self.public_block_size = block_spec.numel
        self.public_dtype = block_spec.dtype

    def allocate_private_blocks(self, block_specs: Iterable[BlockSpec]):
        """
        Allocate private blocks for the memory pool. This method will allocate private blocks according to the block_specs.

        Args:
            block_specs (Iterable[BlockSpec]): the block specs of the private blocks to be allocated
        """
        # allocate private blocks
        assert not self.__private_allocated_flag, 'Private blocks have been allocated to this MemoryPool object, it is not allowed to allocate again.'

        for spec in block_specs:
            block = PrivateBlock(spec.numel, spec.dtype, self.device_type)
            self.private_space += block.size_in_bytes
            self.private_blocks.append(block)
            self.private_lookup_dict[spec].append(block)

        self.__private_allocated_flag = True

    def __repr__(self) -> str:
        return f'Memory Pool(\n\tpublic_space = {_format_memory(self.public_space)}, \n\tprivate_space={_format_memory(self.private_space)}\n)'

    def get_private_block(self, numel: int, dtype: torch.dtype) -> PrivateBlock:
        """
        Get a private block with the given numel and dtype.
        """
        block_list = self.private_lookup_dict.get(BlockSpec(numel=numel, dtype=dtype))

        if len(block_list) == 0:
            raise ValueError(f'No private block with numel={numel} and dtype={dtype} is found.')
        else:
            return block_list.pop()

    def pop_public_block(self) -> PublicBlock:
        """
        Get a public block from the memory pool.
        """
        self.public_free_count -= 1
        self.public_used_count += 1

        block = self.public_free_blocks.pop()
        self.public_used_blocks.add(block)
        return block

    def free_public_block(self, block: TensorBlock) -> PublicBlock:
        """
        Free a public block to the memory pool.

        Args:
            block (TensorBlock): the public block to be freed
        """
        assert isinstance(block, PublicBlock)
        assert block in self.public_used_blocks, f'Cound not find the given block in the used public blocks'

        # update counter
        self.public_free_count += 1
        self.public_used_count -= 1

        # update free and used blocks
        self.public_used_blocks.remove(block)
        self.public_free_blocks.append(block)

        return block
