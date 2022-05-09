from .utils import InsertPostInitMethodToModuleSubClasses
import torch
from colossalai.tensor import ColoTensor, ColoParameter
import types

from torch import nn
from typing import Iterator, Tuple, Union, Optional

# Adapted from torch.nn.module.Module.register_param
def _register_parameter_with_colotensor(self, name: str, param):
    if '_parameters' not in self.__dict__:
        raise AttributeError(
            "cannot assign parameter before Module.__init__() call")

    if not isinstance(name, torch._six.string_classes):
        raise TypeError("parameter name should be a string. "
                        "Got {}".format(torch.typename(name)))
    if '.' in name:
        raise KeyError("parameter name can't contain \".\"")
    if name == '':
        raise KeyError("parameter name can't be empty string \"\"")
    if hasattr(self, name) and name not in self._parameters:
        raise KeyError("attribute '{}' already exists".format(name))

    if param is None:
        self._parameters[name] = None
    elif not isinstance(param, (torch.nn.Parameter, ColoParameter)):
        raise TypeError("cannot assign '{}' object to parameter '{}' "
                        "(torch.nn.Parameter or ColoParameter or None required)"
                        .format(torch.typename(param), name))
    elif param.grad_fn:
        raise ValueError(
            "Cannot assign non-leaf Tensor to parameter '{0}'. Model "
            "parameters must be created explicitly. To express '{0}' "
            "as a function of another Tensor, compute the value in "
            "the forward() method.".format(name))
    else:
        self._parameters[name] = param

# Adapted from torch.nn.module.Module.__setattr__
def _setattr_with_colotensor(self, name: str, value: Union[torch.Tensor, torch.nn.Module, ColoTensor]):
    def remove_from(*dicts_or_sets):
        for d in dicts_or_sets:
            if name in d:
                if isinstance(d, dict):
                    del d[name]
                else:
                    d.discard(name)

    params = self.__dict__.get('_parameters')
    if isinstance(value, (ColoTensor, torch.nn.Parameter)):
        if params is None:
            raise AttributeError(
                "cannot assign parameters before Module.__init__() call")
        remove_from(self.__dict__, self._buffers, self._modules, self._non_persistent_buffers_set)
        self.register_parameter(name, value)
    elif params is not None and name in params:
        if value is not None:
            raise TypeError("cannot assign '{}' as parameter '{}' "
                            "(torch.nn.Parameter or None expected)"
                            .format(torch.typename(value), name))
        self.register_parameter(name, value)
    else:
        modules = self.__dict__.get('_modules')
        if isinstance(value, torch.nn.Module):
            if modules is None:
                raise AttributeError(
                    "cannot assign module before Module.__init__() call")
            remove_from(self.__dict__, self._parameters, self._buffers, self._non_persistent_buffers_set)
            modules[name] = value
        elif modules is not None and name in modules:
            if value is not None:
                raise TypeError("cannot assign '{}' as child module '{}' "
                                "(torch.nn.Module or None expected)"
                                .format(torch.typename(value), name))
            modules[name] = value
        else:
            buffers = self.__dict__.get('_buffers')
            if buffers is not None and name in buffers:
                if value is not None and not isinstance(value, torch.Tensor):
                    raise TypeError("cannot assign '{}' as buffer '{}' "
                                    "(torch.Tensor or None expected)"
                                    .format(torch.typename(value), name))
                buffers[name] = value
            else:
                object.__setattr__(self, name, value)

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

        # TODO(jzy) replace it with old __setattr__ in the exit() of context?
        torch.nn.Module.__setattr__ = _setattr_with_colotensor
        torch.nn.Module.register_parameter = _register_parameter_with_colotensor

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

            # detaching tensor is necessary for optimizers.
            requires_grad = param.requires_grad
            tensor_detached = param.to(self._device).detach()
            tensor_detached.requires_grad = requires_grad

            colo_param = ColoParameter.init_from_torch_tensor(tensor=tensor_detached, save_payload=save_torch_payload)
            setattr(module, name, colo_param)

        ColoModulize(module)