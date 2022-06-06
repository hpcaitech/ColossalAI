import torch
import inspect
from colossalai.utils.model.utils import InsertPostInitMethodToModuleSubClasses, call_to_str
from colossalai.builder.pipeline import partition_uniform, partition_balanced
from colossalai.nn.layer.utils import CheckpointModule
from colossalai.tensor import ColoTensor


class PipelinableContext(InsertPostInitMethodToModuleSubClasses):

    def __init__(self):
        super().__init__()
        self._layer_spec_dict = {}
        self._root_children = None
        self._model = None
        self._layer_spec_list = []
        self._func_dict = {}
        self._policy = "balanced"

    @property
    def policy(self):
        return self._policy

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
        module_id = id(module)
        modified_args = []
        for obj in args:
            if issubclass(obj.__class__, torch.nn.modules.module.Module):
                obj = self._layer_spec_dict[id(obj)]
            modified_args.append(obj)

        modified_kwargs = {}
        for k, v in kwargs.items():
            if issubclass(v.__class__, torch.nn.modules.module.Module):
                v = self._layer_spec_dict[id(v)]
            # (lyl)TODO: analyse ColoTensor as well
            modified_kwargs[k] = v

        modified_args = tuple(modified_args)
        self._root_children = list(module.children())
        self._model = module
        layer_spec = LayerSpec(module.__class__, *modified_args, **modified_kwargs)
        layer_spec.set_children(module.children())
        self._layer_spec_dict[module_id] = layer_spec
        name_list = []
        for name, param in module.named_parameters():
            if isinstance(param, ColoTensor):
                continue
            name_list.append((name, param))

        for name, param in name_list:
            delattr(module, name)
            setattr(module, name, ColoTensor.from_torch_tensor(param))

    def to_layer_list(self, exec_seq=None):
        """
        Create a layer spec list and func list with execution sequence given by user.
        If exec_seq is None, we will take the module initizing order as execution order.
        """
        if exec_seq is None:
            # if user do not provide the model executing sequence, we use the initialization order as the executing order.
            children_name = []
            for child in self._root_children:
                layer_spec = self._layer_spec_dict[id(child)]
                if layer_spec.typename in (torch.nn.modules.container.ModuleList,
                                           torch.nn.modules.container.Sequential):
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
            for index, element in enumerate(exec_seq):
                if isinstance(element, str):
                    module = dict(self._model.named_modules())[element]
                    layer_spec = self._layer_spec_dict[id(module)]
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
        Partitioned model will be built respect to partion policy.
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
            else:
                raise ValueError("A string partition policy should be one of ['uniform', 'balanced'].")
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
        pipeline_model = PipelinableModel(module_list_in_partition, front_func_dict_in_partition,
                                          behind_func_dict_in_partition)

        return pipeline_model

    def load_policy(self, policy):
        self._policy = policy


def _build_kwargs_for_module(function, kw_dict):
    """
    Generally, the first argument of module.forward is an input tensor come from the previous layer.
    Therefore, we just filter the kwargs from second element of the dictionary.
    """
    sig = inspect.signature(function)
    if len(sig.parameters) <= 1:
        return None
    args_name_list = list(sig.parameters.keys())
    kw_dict = {k: v for k, v in kw_dict.items() if k in args_name_list[1:]}
    return kw_dict


def _build_kwargs_for_function(function, kw_dict):
    sig = inspect.signature(function)
    kw_dict = {k: v for k, v in kw_dict.items() if k in sig.parameters}
    if len(kw_dict) == 0:
        return None
    return kw_dict


def _exec_func_with_kwargs(func, kw_dict, input_tensor, kwargs):
    """
    We suppose the callable object passed to to_layer_list method in two purpose:
        a. use the callable object to modify input tensor, such as \
            lambda x: torch.flatten(x, 1)
        b. use the callable object to modify kwargs value, such as \
            def foo(attention_mask=None):
                if attention_mask is not None:
                    batch_size = input_ids.shape[0]
                    attention_mask = attention_mask.view(batch_size, -1)
                return attention_mask
    """

    if kw_dict is not None:
        rst = func(**kw_dict)
        if isinstance(rst, tuple):
            for i, k in enumerate(kw_dict.keys()):
                kwargs[k] = rst[i]
        else:
            for k in kw_dict.keys():
                kwargs[k] = rst
        return input_tensor
    return func(input_tensor)


def _exec_funcs_with_kwargs(func_dict, func_key, input_tensor, kwargs):

    assert func_key in func_dict, f"{func_key} is not in the function_dict."
    funcs_to_exec = func_dict[func_key]
    if isinstance(funcs_to_exec, list):
        for f in funcs_to_exec:
            f_kwargs = _build_kwargs_for_function(f, kwargs)
            input_tensor = _exec_func_with_kwargs(f, f_kwargs, input_tensor, kwargs)
    else:
        f_kwargs = _build_kwargs_for_function(funcs_to_exec, kwargs)
        input_tensor = _exec_func_with_kwargs(funcs_to_exec, f_kwargs, input_tensor, kwargs)

    return input_tensor


class PipelinableModel(torch.nn.Module):

    def __init__(self, module_list, front_func_dict, behind_func_dict):
        super().__init__()
        self._module_list = module_list
        self._front_func_dict = front_func_dict
        self._behind_func_dict = behind_func_dict

    def forward(self, input_tensor, **kwargs):

        for module in self._module_list:

            if id(module) in self._front_func_dict:
                input_tensor = _exec_funcs_with_kwargs(self._front_func_dict, id(module), input_tensor, kwargs)

            if isinstance(module, CheckpointModule):
                forward_func = module._forward
            else:
                forward_func = module.forward
            if input_tensor is None:
                module_kwargs = _build_kwargs_for_function(forward_func, kwargs)
            else:
                module_kwargs = _build_kwargs_for_module(forward_func, kwargs)
            if module_kwargs is not None and input_tensor is not None:
                if isinstance(module, CheckpointModule):
                    convert_kwargs_to_args = []
                    for v in module_kwargs.values():
                        convert_kwargs_to_args.append(v)
                    rst = module(input_tensor, *convert_kwargs_to_args)
                else:
                    rst = module(input_tensor, **module_kwargs)
                if isinstance(rst, tuple):
                    input_tensor = rst[0]
                else:
                    input_tensor = rst
            elif module_kwargs is not None and input_tensor is None:
                if isinstance(module, CheckpointModule):
                    convert_kwargs_to_args = []
                    for v in module_kwargs.values():
                        convert_kwargs_to_args.append(v)
                    rst = module(input_tensor, *convert_kwargs_to_args)
                else:
                    rst = module(**module_kwargs)
                if isinstance(rst, tuple):
                    input_tensor = rst[0]
                else:
                    input_tensor = rst
            else:
                input_tensor = module(input_tensor)

            if id(module) in self._behind_func_dict:
                input_tensor = _exec_funcs_with_kwargs(self._behind_func_dict, id(module), input_tensor, kwargs)

        return input_tensor


class LayerSpec:

    def __init__(self, typename, *module_args, **module_kwargs):
        self.typename = typename
        self.module_args = module_args
        self.module_kwargs = module_kwargs
        self.children = None
        self._param_count = 0

        if not issubclass(typename, torch.nn.Module):
            raise RuntimeError('LayerSpec only supports torch.nn.Module types.')

    def __repr__(self):
        return call_to_str(self.typename.__name__, self.module_args, self.module_kwargs)

    @property
    def param_count(self):
        return self._param_count

    def build(self):
        """Build the stored specification."""

        recovered_args = []
        for obj in self.module_args:
            if isinstance(obj, LayerSpec):
                obj = obj.build()
            recovered_args.append(obj)
        recovered_args = tuple(recovered_args)

        recovered_kwargs = {}
        for k, v in self.module_kwargs.items():
            if isinstance(v, LayerSpec):
                v = v.build()
            recovered_kwargs[k] = v

        return self.typename(*recovered_args, **recovered_kwargs)

    def set_children(self, children):
        self.children = children

    def count_params(self):
        self._param_count = 0
        layer = self.build()
        for param in layer.parameters():
            self._param_count += param.numel()
        return self._param_count

    def reset_param_count(self):
        self._param_count = 0
