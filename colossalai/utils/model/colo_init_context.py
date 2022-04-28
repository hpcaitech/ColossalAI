from .utils import InsertPostInitMethodToModuleSubClasses
import torch
from colossalai.tensor import ColoTensor
import types

from torch import nn
from typing import Iterator, Tuple, Union


def ColoModulize(module):
    """
    Replacing the parameters() and named_parameters() with our customized ones
    """

    def named_params_with_colotensor(
        module: nn.Module,
        prefix: str = '',
        recurse: bool = True,
    ) -> Iterator[Tuple[str, Union[nn.Parameter, ColoTensor]]]:
        modules = module.named_modules(prefix=prefix) if recurse else [(prefix, module)]

        memo = set()
        for mod_prefix, mod in modules:
            # find all colotensors tensor params
            for name, val in vars(mod).items():
                if isinstance(val, ColoTensor) and val not in memo:
                    memo.add(val)
                    name = mod_prefix + ('.' if mod_prefix else '') + name
                    yield name, val

        # find all nn.Parameters
        for name, val in module.old_named_parameters(recurse=recurse):
            yield name, val

    def fake_parameters(self, *args, **kargs):
        for name, p in named_params_with_colotensor(self, *args, **kargs):
            if isinstance(p, ColoTensor):
                yield p.torch_tensor()
            elif isinstance(p, torch.Tensor):
                yield p

    def fake_named_parameters(self, *args, **kargs):
        for name, p in named_params_with_colotensor(self, *args, **kargs):
            if isinstance(p, ColoTensor):
                yield name, p.torch_tensor()
            elif isinstance(p, torch.Tensor):
                yield name, p

    def colo_parameters(self, *args, **kargs):
        for _, p in named_params_with_colotensor(self, *args, **kargs):
            yield p

    def colo_named_parameters(self, *args, **kargs):
        for name, p in named_params_with_colotensor(self, *args, **kargs):
            yield name, p

    module.old_named_parameters = module.named_parameters
    module.old_parameters = module.parameters

    funcType = types.MethodType
    module.parameters = funcType(fake_parameters, module)
    module.named_parameters = funcType(fake_named_parameters, module)
    module.colo_parameters = funcType(colo_parameters, module)
    module.colo_named_parameters = funcType(colo_named_parameters, module)
    module._colo_visited = True


class ColoInitContext(InsertPostInitMethodToModuleSubClasses):

    def __init__(self, lazy_memory_allocate: bool = False, device: torch.device = torch.device('cpu')):
        """
        Args:
            lazy_memory_allocate (bool, optional): whether to allocate memory for the parameter tensors. Defaults to False.
            device (torch.device, optional): the device parameters initialized are resident on. Defaults to torch.device('cpu').
        """
        super().__init__()
        self._lazy_memory_allocate = lazy_memory_allocate
        self._device = device

    def _post_init_method(self, module: torch.nn.Module, *args, **kwargs):
        """
        The function to call at the end of the constructor of each module.
        FIXME(fjr) The module may be passed to this function multiple times?
        """
        if hasattr(module, '_colo_visited'):
            return

        name_list = []
        for name, param in module.named_parameters(recurse=False):
            if isinstance(param, ColoTensor):
                continue
            name_list.append((name, param))

        save_torch_payload = True if not self._lazy_memory_allocate else False
        for name, param in name_list:
            delattr(module, name)
            setattr(
                module, name,
                ColoTensor.init_from_torch_tensor(tensor=param.to(self._device),
                                                  save_payload=save_torch_payload,
                                                  is_model_data=True))

        ColoModulize(module)
