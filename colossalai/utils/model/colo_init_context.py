from typing import Dict, Iterator, Optional, Tuple, Union

import torch
from torch import nn

from colossalai.nn.parallel.layers import ColoEmbedding, ColoLinear, register_colo_module
from colossalai.tensor import ColoParameter, ColoTensor, ProcessGroup, ShardSpec

from .utils import InsertPostInitMethodToModuleSubClasses

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


class ColoInitContext(InsertPostInitMethodToModuleSubClasses):

    def __init__(self,
                 device: torch.device = torch.device('cpu'),
                 dtype: torch.dtype = torch.float,
                 default_pg: Optional[ProcessGroup] = None,
                 default_dist_spec=None):
        """
        Args:
            device (torch.device): the device where parameters initialized are resident. Defaults to torch.device('cpu').
            dtype (torch.dtype): the dtype of parameters initialized. Defults to torch.float.
            default_pg (ProcessGroup): the default process group for all initialized parameters.
            default_dist_spec: the default distributed specifications.
        """
        super().__init__()
        self._device = device
        self._dtype = dtype

        self._register_colo_modules()
        self._default_pg = default_pg
        self._default_dist_spec = default_dist_spec

    def _register_colo_modules(self):
        register_colo_module(torch.nn.Linear, ColoLinear())
        register_colo_module(torch.nn.Embedding, ColoEmbedding())

    def _pre_context_exec(self):
        pass

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
                # detaching tensor is necessary for optimizers.
                requires_grad = param.requires_grad

                # param is the global tensor.
                colo_param = ColoParameter(param.to(device=self._device, dtype=self._dtype),
                                           requires_grad=requires_grad)

                # if default_shard_plan exists, shard the param during initialization.
                # This can reduce the model size after initialization.
                # NOTE() embedding usually can not be correctly sharded. So I use except to handle
                # the param that can not be sharded by the default plan
                if self._default_pg is not None:
                    colo_param.set_process_group(self._default_pg)

                if self._default_dist_spec is not None:
                    try:
                        colo_param.set_dist_spec(self._default_dist_spec)
                    except:
                        pass

                replaced_tensors[param] = colo_param
            delattr(submodule, param_name)
            setattr(submodule, param_name, colo_param)
            colo_param.shared_param_modules.append(submodule)

        module.to(self._device)
        ColoModulize(module)
