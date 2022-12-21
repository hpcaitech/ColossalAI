from time import time
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from colossalai.gemini.chunk import ChunkManager
from colossalai.gemini.chunk.search_utils import in_ddp, search_chunk_configuration
from colossalai.gemini.memory_tracer import MemStats


def init_chunk_manager(model: nn.Module,
                       init_device: Optional[torch.device] = None,
                       hidden_dim: Optional[int] = None,
                       search_range_mb: Optional[float] = None,
                       min_chunk_size_mb: Optional[float] = None,
                       filter_exlarge_params: Optional[bool] = None) -> ChunkManager:

    kwargs_dict = dict()

    if hidden_dim:
        search_interval_byte = hidden_dim
    else:
        search_interval_byte = 1024    # 1kb
    kwargs_dict["search_interval_byte"] = search_interval_byte

    if search_range_mb:
        kwargs_dict["search_range_mb"] = search_range_mb

    if min_chunk_size_mb:
        kwargs_dict["min_chunk_size_mb"] = min_chunk_size_mb

    if filter_exlarge_params:
        kwargs_dict["filter_exlarge_params"] = filter_exlarge_params

    params_sizes = [p.numel() for p in model.parameters() if in_ddp(p)]
    total_size = sum(params_sizes) / 1024**2

    dist.barrier()
    begin = time()

    config_dict, wasted_size = search_chunk_configuration(model, **kwargs_dict)

    dist.barrier()
    end = time()
    span_s = end - begin
    wasted_size /= 1024**2

    if dist.get_rank() == 0:
        print("searching chunk configuration is completed in {:.2f} s.\n".format(span_s),
              "used number: {:.2f} MB, wasted number: {:.2f} MB\n".format(total_size, wasted_size),
              "total wasted percentage is {:.2f}%".format(100 * wasted_size / (total_size + wasted_size)),
              sep='',
              flush=True)
    dist.barrier()

    chunk_manager = ChunkManager(config_dict, init_device)
    return chunk_manager
