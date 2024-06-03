import torch

from colossalai.legacy.context import ParallelMode
from colossalai.legacy.core import global_context as gpc
from colossalai.legacy.nn.layer.utils import CheckpointModule
from colossalai.tensor import ColoParameter
from colossalai.utils.model.utils import InsertPostInitMethodToModuleSubClasses

from .layer_spec import LayerSpec
from .utils import (
    build_kwargs_for_module,
    call_module,
    customized_partition,
    exec_funcs_with_kwargs,
    partition_balanced,
    partition_uniform,
)


class PipelinableContext(InsertPostInitMethodToModuleSubClasses):
    """
    A context manager to split the model into pipeline stages.
    """

    def __init__(self, policy: str = "balanced"):
        super().__init__()
        self._layer_spec_dict = {}
        self._root_children = None
        self._model = None
        self._layer_spec_list = []
        self._func_dict = {}
        self._policy = policy

    @property
    def policy(self):
        return self._policy

    @policy.setter
    def policy(self, policy: str):
        self._policy = policy

    @property
    def layers_count(self):
        return len(self._layer_spec_list)

    @property
    def funcs_count(self):
        return len(self._func_dict)

    def _pre_context_exec(self):
        """
        The Callback function when entering the context
        """
        # reserve rng states
        self.cpu_rng_state = torch.get_rng_state()
        self.cuda_rng_state = torch.cuda.get_rng_state()

    def _post_context_exec(self):
        """
        The callback function when exiting context.
        """

        # reset rng states
        torch.set_rng_state(self.cpu_rng_state)
        torch.cuda.set_rng_state(self.cuda_rng_state)

    def _post_init_method(self, module: torch.nn.Module, *args, **kwargs):
        """
        The function to call at the end of the constructor of each module.
        NOTE() The module may be passed to this function multiple times.
        """
        # iterate over the positional arguments
        # to check if an argument is a torch Module
        # if found any torch Module, replace it with its layer spec
        # for storage purpose
        modified_args = []
        for arg in args:
            if isinstance(arg, torch.nn.Module):
                # if nn.Module is an argument of a non-root module, then we should convert it to layer spec, which make sure the correct init method used in the real build.
                # if nn.Module is an argument of the root module, then we should just record the module instance itself, because those instance has been built outside of the context.
                if id(arg) in self._layer_spec_dict:
                    arg = self._layer_spec_dict[id(arg)]

            modified_args.append(arg)

        # to the same for the keyword arguments
        modified_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.nn.Module):
                v = self._layer_spec_dict[id(v)]
            # (lyl)TODO: analyze ColoTensor as well
            modified_kwargs[k] = v

        # keep track of the module children
        # as torch.nn.Module.__init__ is called from inner module to outer module,
        # the final value of self._model will be the outermost model
        # e.g. if the model is torchvision.models.resnet18, then the final value of self._model
        # will be the ``ResNet`` object.
        self._root_children = list(module.children())
        self._model = module

        # store the children to keep the module hierarchy
        layer_spec = LayerSpec(module.__class__, *modified_args, **modified_kwargs)
        layer_spec.set_children(module.children())

        # store the layer spec in this context
        module_id = id(module)
        self._layer_spec_dict[module_id] = layer_spec

        # convert all torch.nn.Parameter to colossalai.tensor.ColoParameter
        name_list = []
        for name, param in module.named_parameters():
            if isinstance(param, ColoParameter):
                continue
            name_list.append((name, param))

        for name, param in name_list:
            if hasattr(module, name):
                delattr(module, name)
            setattr(module, name, ColoParameter.from_torch_tensor(tensor=param.data, requires_grad=param.requires_grad))

    def to_layer_list(self, exec_seq=None):
        """
        Create a layer spec list and func list with execution sequence given by user.
        If exec_seq is None, we will take the module initializing order as execution order.
        """

        self._exec_seq = exec_seq
        if exec_seq is None:
            # if user do not provide the model executing sequence, we use the initialization order as the executing order.
            children_name = []
            for child in self._root_children:
                layer_spec = self._layer_spec_dict[id(child)]
                if layer_spec.typename in (
                    torch.nn.modules.container.ModuleList,
                    torch.nn.modules.container.Sequential,
                ):
                    for child_in_container in layer_spec.children:
                        self._layer_spec_list.append(self._layer_spec_dict[id(child_in_container)])
                        for name, module in self._model.named_modules():
                            if id(module) == id(child_in_container):
                                children_name.append(name)
                                break
                else:
                    self._layer_spec_list.append(layer_spec)
                    for name, module in self._model.named_modules():
                        if id(module) == id(child):
                            children_name.append(name)
                            break

        else:
            front_funcs_list = []
            named_modules = dict(self._model.named_modules())
            for index, element in enumerate(exec_seq):
                if isinstance(element, str):
                    if element == "SPLIT_NODE":
                        continue
                    assert (
                        element in named_modules
                    ), f"Found invalid module name {element}, please check if you spell the module name correctly."

                    # get the layer spec based on the module ID
                    module = named_modules[element]
                    layer_spec = self._layer_spec_dict[id(module)]

                    # check whether there are functions which should be executed before this module
                    if len(front_funcs_list) != 0:
                        func_key = (layer_spec, "front")
                        if func_key not in self._func_dict:
                            self._func_dict[func_key] = []
                        for f in front_funcs_list:
                            self._func_dict[func_key].append(f)
                        front_funcs_list = []

                    func_key = (layer_spec, "behind")
                    self._layer_spec_list.append(layer_spec)
                elif isinstance(element, tuple) and element[1] == "front":
                    front_funcs_list.append(element[0])
                else:
                    if func_key not in self._func_dict:
                        self._func_dict[func_key] = []
                    if isinstance(element, tuple):
                        self._func_dict[func_key].append(element[0])
                    else:
                        self._func_dict[func_key].append(element)

    def partition(self, num_chunks, pipeline_size, rank):
        """
        Partitioned model will be built respect to partition policy.
        The real module instance will be built in this method.
        """
        if isinstance(self._policy, str):
            if self._policy == "uniform":
                parts = partition_uniform(len(self._layer_spec_list), pipeline_size, num_chunks)[rank]
            elif self._policy == "balanced":
                param_counts = []
                for layer_spec in self._layer_spec_list:
                    param_counts.append(layer_spec.count_params())
                parts = partition_balanced(param_counts, pipeline_size, num_chunks)[rank]
            elif self._policy == "customized":
                assert (
                    self._exec_seq is not None
                ), f"An explicit exec_seq must be defined by user in customized policy mode."
                self.customized_parts = customized_partition(self._exec_seq)
                assert len(self.customized_parts) == gpc.get_world_size(
                    ParallelMode.PIPELINE
                ), f"World size is {gpc.get_world_size(ParallelMode.PIPELINE)}, but the number of partitions is {len(self.customized_parts)}"
                parts = self.customized_parts[rank]
            else:
                raise ValueError("A string partition policy should be one of ['uniform', 'balanced', 'customized'].")
        elif isinstance(self._policy, dict):
            parts = self._policy[rank]
        else:
            raise ValueError("A partition policy should be either a string or a dictionary.")

        layers_to_build = []
        for start, end in parts:
            layers_to_build += self._layer_spec_list[start:end]
        behind_func_dict_in_partition = {}
        front_func_dict_in_partition = {}
        module_list_in_partition = []
        for layer in layers_to_build:
            module = layer.build()
            module_list_in_partition.append(module)
            if (layer, "front") in self._func_dict:
                front_func_dict_in_partition[id(module)] = self._func_dict[(layer, "front")]
            elif (layer, "behind") in self._func_dict:
                behind_func_dict_in_partition[id(module)] = self._func_dict[(layer, "behind")]
        module_list_in_partition = torch.nn.ModuleList(module_list_in_partition)
        pipeline_model = PipelinableModel(
            module_list_in_partition, front_func_dict_in_partition, behind_func_dict_in_partition
        )

        return pipeline_model


class PipelinableModel(torch.nn.Module):
    def __init__(self, module_list, front_func_dict, behind_func_dict):
        super().__init__()
        self._module_list = module_list
        self._front_func_dict = front_func_dict
        self._behind_func_dict = behind_func_dict

    def forward(self, *input_tensor, **kwargs):
        for module in self._module_list:
            if id(module) in self._front_func_dict:
                input_tensor = exec_funcs_with_kwargs(self._front_func_dict, id(module), input_tensor, kwargs)

            if isinstance(module, CheckpointModule):
                forward_func = module._forward
            else:
                forward_func = module.forward
            module_kwargs = build_kwargs_for_module(forward_func, input_tensor, kwargs)
            if input_tensor is None:
                input_tensor = call_module(module, kwargs=module_kwargs)
            elif isinstance(input_tensor, torch.Tensor):
                input_tensor = call_module(module, args=(input_tensor,), kwargs=module_kwargs)
            else:
                input_tensor = call_module(module, args=input_tensor, kwargs=module_kwargs)

            if id(module) in self._behind_func_dict:
                input_tensor = exec_funcs_with_kwargs(self._behind_func_dict, id(module), input_tensor, kwargs)

        return input_tensor
