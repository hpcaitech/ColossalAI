from .utils import InsertPostInitMethodToModuleSubClasses
import torch
# from colossalai.logging import get_dist_logger
from colossalai.tensor import ColoTensor

# _orig_torch_empty = torch.empty


class ColoInitContext(InsertPostInitMethodToModuleSubClasses):

    def __init__(self, lazy_memory_allocate=False):
        super().__init__()
        self._lazy_memory_allocate = lazy_memory_allocate

    def _post_init_method(self, module: torch.nn.Module, *args, **kwargs):
        """
        The function to call at the end of the constructor of each module.
        FIXME(fjr) The module may be passed to this function multiple times?
        """
        name_list = []
        for name, param in module.named_parameters():
            if isinstance(param, ColoTensor):
                continue
            name_list.append((name, param))

        save_torch_payload = True if not self._lazy_memory_allocate else False
        for name, param in name_list:
            delattr(module, name)
            setattr(module, name, ColoTensor.init_from_torch_tensor(tensor=param.data, save_payload=save_torch_payload))
