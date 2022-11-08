import torch

from colossalai.gemini.chunk import init_chunk_manager
from colossalai.gemini.gemini_mgr import GeminiManager

from .data_parallel import ZeroDDP


class GeminiDDP(ZeroDDP):

    def __init__(self,
                 module: torch.nn.Module,
                 device: torch.device,
                 placement_policy: str = "cpu",
                 pin_memory: bool = False,
                 force_outputs_fp32: bool = False,
                 search_range_mb: int = 32) -> None:
        """
        A torch.Module warpper using ZeRODPP and Genimi.
        ZeRO is for parallel. Gemini is for memory management.

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
        """
        chunk_manager = init_chunk_manager(model=module, init_device=device, search_range_mb=search_range_mb)
        gemini_manager = GeminiManager(placement_policy, chunk_manager, module)
        super().__init__(module, gemini_manager, pin_memory, force_outputs_fp32)
