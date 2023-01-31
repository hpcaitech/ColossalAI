from typing import Optional

import torch

from colossalai.gemini.chunk import init_chunk_manager
from colossalai.gemini.gemini_mgr import GeminiManager
from colossalai.gemini.memory_tracer import MemStats

from .data_parallel import ZeroDDP


class GeminiDDP(ZeroDDP):

    def __init__(self,
                 module: torch.nn.Module,
                 device: torch.device,
                 placement_policy: str = "cpu",
                 pin_memory: bool = False,
                 force_outputs_fp32: bool = False,
                 strict_ddp_mode: bool = False,
                 search_range_mb: int = 32,
                 hidden_dim: Optional[int] = None,
                 min_chunk_size_mb: float = 32,
                 memstats: Optional[MemStats] = None) -> None:
        """
        A torch.Module warpper using ZeRO-DP and Genimi.
        ZeRO is for parallel. Gemini is for memory management.
        WARNING: The class will modify the module inline!

        Example:
            model is initialized under the context of ColoInitContext
            >>> model = GeminiDDP(model, torch.cuda.current_device(), "cuda")
            >>> logits = model(x)
            >>> loss = criterion(logits, labels)
            >>> model.backward(loss)

        Args:
            module (torch.nn.Module): the model to be wrapped.
            device (torch.device): device to place the model.
            placement_policy (str, optional): "cpu", "cuda", "auto". Defaults to "cpu".
            pin_memory (bool, optional): use pin memory on CPU. Defaults to False.
            force_outputs_fp32 (bool, optional): force outputs are fp32. Defaults to False.
            search_range_mb (int, optional): chunk size searching range in MegaByte. Defaults to 32.
            hidden_dim (int, optional): the hidden dimension of DNN.
                Users can provide this argument to speed up searching.
                If users do not know this argument before training, it is ok. We will use a default value 1024.
            min_chunk_size_mb (float, optional): the minimum chunk size in MegaByte.
                If the aggregate size of parameters is still samller than the minimum chunk size,
                all parameters will be compacted into one small chunk.
            memstats (MemStats, optional) the memory statistics collector by a runtime memory tracer.
        """
        # some ugly hotfix for the compatibility with Lightning
        if search_range_mb is None:
            search_range_mb = 32

        chunk_manager = init_chunk_manager(model=module,
                                           init_device=device,
                                           hidden_dim=hidden_dim,
                                           search_range_mb=search_range_mb,
                                           min_chunk_size_mb=min_chunk_size_mb,
                                           strict_ddp_flag=strict_ddp_mode)
        gemini_manager = GeminiManager(placement_policy, chunk_manager, memstats)
        super().__init__(module, gemini_manager, pin_memory, force_outputs_fp32, strict_ddp_mode)
