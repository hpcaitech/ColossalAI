from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class DrafterOutput:
    """
    Dataclass for drafter model outputs.

    Args:
        speculated_length (int): Speculated length of the output sequence
            It is always less than or equal to spec_num during drafter's speculation process
        logits (torch.FloatTensor): Logits of the output sequence
        next_tokens (torch.Tensor): Next token ids
        past_key_values (Optional[Tuple[Tuple[torch.FloatTensor]]]): Past key values of the output sequence
    """

    speculated_length: int = None
    logits: torch.FloatTensor = None
    next_tokens: torch.Tensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None

    def __post_init__(self):
        assert self.speculated_length is not None and self.speculated_length >= 0
        if self.past_key_values is not None:
            assert isinstance(self.past_key_values, tuple), "Past key values should be a tuple"
            assert all([isinstance(past_key_value, tuple) for past_key_value in self.past_key_values])


@dataclass
class GlideInput:
    """Dataclass for Glide Models (e.g. `colossalai/inference/modeling/models/glide_llama.py`).
    Used for pack data that will be used during glimpsing KV Caches of the main model.

    Args:
        block_tables (torch.Tensor): [num_seqs, max_blocks_per_seq] The block table of KV Caches.
        large_k_cache (torch.Tensor): [num_blocks, num_kv_heads, block_size, head_size]
            Blocked key cache of the main model
        large_v_cache (torch.Tensor): Blocked value cache of the main model. It has the same shape as k cache.
        sequence_lengths (torch.Tensor): [num_seqs] Sequence lengths of the current batch.
    """

    block_tables: torch.Tensor = None
    large_k_cache: torch.Tensor = None
    large_v_cache: torch.Tensor = None
    sequence_lengths: torch.Tensor = None
    n_spec_tokens: int = 5

    @property
    def glimpse_ready(self):
        return all(
            attr is not None
            for attr in [self.block_tables, self.large_k_cache, self.large_v_cache, self.sequence_lengths]
        )
