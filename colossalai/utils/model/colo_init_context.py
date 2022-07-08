from .utils import InsertPostInitMethodToModuleSubClasses
import torch
from colossalai.tensor import ColoTensor, ColoParameter, distspec, ProcessGroup

from colossalai.nn.parallel.layers import register_colo_module, \
    ColoLinear, ColoEmbedding
from copy import copy
from torch import nn
from typing import Iterator, Tuple, Union
from functools import partialmethod
# find named_params includes replica


def _named_params_with_replica(
    module: nn.Module,
    prefix: str = '',
    recurse: bool = True,
) -> Iterator[Tuple[str, Union[nn.Parameter, ColoTensor]]]:
    modules = module.named_modules(prefix=prefix) if recurse else [(prefix, module)]

    for mod_prefix, mod in modules:
        for name, val in mod._parameters.items():
            if val is None:
                continue
            name = mod_prefix + ('.' if mod_prefix else '') + name
            yield name, val


def ColoModulize(module):
    """
    Replacing the parameters() and named_parameters() with our customized ones
    """

    module._colo_visited = True


def colo_state_dict(self, destination=None, prefix='', keep_vars=False, state_dict_func=None):
    # build param to spec mapping
    mapping1 = dict()
    mapping2 = dict()
    mapping3 = dict()
    # gather all params
    has_dist_parameter = False
    with torch.no_grad():
        for param in self.parameters():
            if isinstance(param, ColoParameter):
                has_dist_parameter = True
                mapping1[id(param)] = copy(param.dist_spec)
                mapping2[id(param)] = copy(param.compute_spec)
                # TODO(jiaruifang) fixme, we should elegently handle the default PG in init context
                if param.get_process_group() is None:
                    param.process_group = ProcessGroup()
                param.set_dist_spec(distspec.replicate())
                mapping3[id(param)] = param.get_process_group()
                param.process_group = None

    # TODO: fix when keep_vars = True
    # when keep_vars = False, the state_dict_func will call detach to create
    # new tensors, but when keep_vars = True, the recovery of spec will be reflected
    # in the `ret`, such that the final state dict will still contain process group,
    # raising exception as it is not serializable
    assert not (keep_vars and has_dist_parameter), 'keep_vars cannot be True when there are distributed ColoParameters.'

    ret = state_dict_func(self, destination, prefix, keep_vars)

    # recover
    with torch.no_grad():
        for param in self.parameters():
            param_id = id(param)
            if param_id in mapping1:
                dist_spec = mapping1[id(param)]
                compute_spec = mapping2[id(param)]
                param.process_group = mapping3[id(param)]
                param.set_tensor_spec(dist_spec, compute_spec)
    return ret


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

        self._register_colo_modules()

    def _register_colo_modules(self):
        register_colo_module(torch.nn.Linear, ColoLinear())
        register_colo_module(torch.nn.Embedding, ColoEmbedding())

    def _pre_context_exec(self):
        self.state_dict_func = nn.Module.state_dict
        nn.Module.state_dict = partialmethod(colo_state_dict, state_dict_func=self.state_dict_func)

    def _post_init_method(self, module: torch.nn.Module, *args, **kwargs):
        """
        The function to call at the end of the constructor of each module.
        FIXME(fjr) The module may be passed to this function multiple times?
        """

        if hasattr(module, '_colo_visited'):
            return

        name_list = []
        for name, param in _named_params_with_replica(module):
            if isinstance(param, ColoTensor):
                continue

            split = name.rfind('.')
            if split >= 0:    # param in submodule
                module_name = name[:split]
                param_name = name[split + 1:]
            else:
                module_name = ''    # param in current module
                param_name = name
            name_list.append((module_name, param_name))

        replaced_tensors = dict(
        )    # record mapping between (torch.Tensor, ColoTensor) to distinguish the same reference
        for module_name, param_name in name_list:
            submodule = module.get_submodule(module_name)
            param = submodule.get_parameter(param_name)
            if param in replaced_tensors:
                colo_param = replaced_tensors[param]
            else:
                save_torch_payload = True if not self._lazy_memory_allocate else False
                # detaching tensor is necessary for optimizers.
                requires_grad = param.requires_grad
                # TODO(jiaruifang) we initialize a Default PG memory
                colo_param = ColoParameter(param.to(self._device), requires_grad=requires_grad)
                # add mapping record
                replaced_tensors[param] = colo_param
            delattr(submodule, param_name)
            setattr(submodule, param_name, colo_param)
            colo_param.shared_param_modules.append(submodule)

        module.to(self._device)
        ColoModulize(module)
