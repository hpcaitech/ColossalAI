from time import time
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from colossalai.gemini.chunk import ChunkManager
from colossalai.gemini.chunk.search_utils import search_chunk_configuration
from colossalai.utils import is_ddp_ignored


def safe_div(a, b):
    if a == 0:
        return 0
    return a / b


def init_chunk_manager(model: nn.Module,
                       init_device: Optional[torch.device] = None,
                       hidden_dim: Optional[int] = None,
                       **kwargs) -> ChunkManager:
    if hidden_dim:
        search_interval_byte = hidden_dim
    else:
        search_interval_byte = 1024    # defaults to 1kb
    kwargs["search_interval_byte"] = search_interval_byte

    dist.barrier()
    begin = time()

    config_dict, total_size, wasted_size = search_chunk_configuration(model, **kwargs)

    dist.barrier()
    end = time()
    span_s = end - begin
    mb_size = 1024**2
    total_size /= mb_size
    wasted_size /= mb_size

    if dist.get_rank() == 0:
        print("searching chunk configuration is completed in {:.2f} s.\n".format(span_s),
              "used number: {:.2f} MB, wasted number: {:.2f} MB\n".format(total_size, wasted_size),
              "total wasted percentage is {:.2f}%".format(100 * safe_div(wasted_size, total_size + wasted_size)),
              sep='',
              flush=True)
    dist.barrier()

    chunk_manager = ChunkManager(config_dict, init_device)
    return chunk_manager
