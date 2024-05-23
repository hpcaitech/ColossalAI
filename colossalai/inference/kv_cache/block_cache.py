from typing import Any

__all__ = ["CacheBlock"]


class CacheBlock:
    """A simplified version of logical cache block used for Paged Attention."""

    def __init__(self, block_id: int, block_size: int, elem_size: int, k_ptrs: Any = None, v_ptrs: Any = None):
        # Unique id of a cache block
        self.block_id = block_id

        # size/capacity of the block in terms of the number of tokens it can hold
        self.block_size = block_size

        # element size in bytes
        self.elem_size = elem_size

        # For common cases, we track the relationships between logical and physical caches in KV Cache Manager,
        # Additionally, k, v pointers can be optionally used for tracking the physical cache by CacheBlock itself.
        self.k_ptrs = k_ptrs
        self.v_ptrs = v_ptrs

        self.ref_count = 0
        # the number of slots that have been allocated (i.e. the number of tokens occupying the block)
        self.allocated_size = 0
        # the token ids whose KV Cache would be written to corresponding physical caches
        # TODO add logics to update token_ids
        self.token_ids = [None] * self.block_size

    @property
    def available_space(self) -> int:
        # `allocated_size` is ensured to be less than or equal to `block_size`
        return self.block_size - self.allocated_size

    def add_ref(self) -> None:
        self.ref_count += 1

    def remove_ref(self) -> None:
        assert self.ref_count > 0, f"Block#{self.block_id} has no reference to remove."
        self.ref_count -= 1

    def has_ref(self) -> bool:
        return self.ref_count > 0

    def allocate(self, size: int) -> None:
        assert size <= self.available_space, f"Block#{self.block_id} has no available space to allocate."
        self.allocated_size += size

    def is_empty(self):
        return self.allocated_size < 1

    def clear(self) -> None:
        self.ref_count = 0
        self.allocated_size = 0

    def __repr__(self):
        return f"CacheBlock#{self.block_id}(ref#{self.ref_count}, allocated#{self.allocated_size})"
