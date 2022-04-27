from .utils import InsertPostInitMethodToModuleSubClasses
import torch
from colossalai.tensor import ColoTensor
import types

from torch import nn
from typing import Iterator, Tuple, Union


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
        name_list = []
        for name, param in module.named_parameters():
            if isinstance(param, ColoTensor):
                continue
            name_list.append((name, param))

        save_torch_payload = True if not self._lazy_memory_allocate else False
        for name, param in name_list:
            delattr(module, name)
            setattr(module, name,
                    ColoTensor.init_from_torch_tensor(tensor=param.to(self._device), save_payload=save_torch_payload))


def ColoModulize(module):
    """
    Replacing the parameters() and named_parameters() with our customized ones
    """

    def fake_parameters(self, *args):
        for name, p in named_params_with_colotensor(self):
            if isinstance(p, ColoTensor):
                yield p.torch_tensor()
            elif isinstance(p, torch.Tensor):
                yield p

    def fake_named_parameters(self, *args):
        for name, p in named_params_with_colotensor(self):
            if isinstance(p, ColoTensor):
                yield name, p.torch_tensor()
            elif isinstance(p, torch.Tensor):
                yield name, p

    def named_params_with_colotensor(
        module: nn.Module,
        prefix: str = '',
        recurse: bool = True,
    ) -> Iterator[Tuple[str, Union[nn.Parameter, ColoTensor]]]:
        modules = module.named_modules(prefix=prefix) if recurse else [(prefix, module)]

        memo = set()
        for mod_prefix, mod in modules:
            # find all sharded tensor params
            for name, val in vars(mod).items():
                if isinstance(val, ColoTensor) and val not in memo:
                    memo.add(val)
                    name = mod_prefix + ('.' if mod_prefix else '') + name
                    yield name, val

        # find all nn.Parameters
        for name, val in module.old_named_parameters():
            yield name, val

    for submodule in module.modules():
        # replacing the parameters() member function with ours
        funcType = types.MethodType
        submodule.old_named_parameters = submodule.named_parameters
        submodule.old_parameters = submodule.parameters

        submodule.parameters = funcType(fake_parameters, submodule)
        submodule.named_parameters = funcType(fake_named_parameters, submodule)


def DeColoModulize(module: nn.Module):
    for submodule in module.modules():
        submodule.named_parameters = submodule.old_named_parameters
        submodule.parameters = submodule.old_parameters
