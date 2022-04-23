import torch
import functools
from colossalai.utils.model.utils import _substitute_init_recursively, InsertPostInitMethodToModuleSubClasses, call_to_str
from colossalai.builder.pipeline import partition_uniform, partition_balanced
from colossalai.core import global_context as gpc


class Pipelinable(InsertPostInitMethodToModuleSubClasses):

    def __init__(self):
        super().__init__()
        self.layer_spec_dict = {}
        self.root_id = None
        self.root_children = None
        self.model = None
        self.layer_spec_list = []
        self.func_dict = {}
        self.policy = "balanced"

    def __enter__(self):
        r"""
        Enter the context scope.
        """

        def preprocess_after(f):

            @functools.wraps(f)
            def wrapper(module: torch.nn.Module, *args, **kwargs):
                f(module, *args, **kwargs)
                self._post_init_method(module, *args, **kwargs)

            return wrapper

        def _enable_class(cls):
            cls._old_init = cls.__init__
            cls.__init__ = preprocess_after(cls.__init__)

        # The function is called during init subclass.
        def _init_subclass(cls, **kwargs):
            cls.__init__ = preprocess_after(cls.__init__)

        # Replace .__init__() for all existing subclasses of torch.nn.Module
        # Excution self._post_init_method after the default init function.
        _substitute_init_recursively(torch.nn.modules.module.Module, _enable_class)

        # holding on to the current __init__subclass__ for exit
        torch.nn.modules.module.Module._old_init_subclass = (torch.nn.modules.module.Module.__init_subclass__)
        # Replace .__init__() for future subclasses of torch.nn.Module
        torch.nn.modules.module.Module.__init_subclass__ = classmethod(_init_subclass)

        self._pre_context_exec()

    def _pre_context_exec(self):
        """ 
        The Callback function when entering the context
        """

        # reserve rng states
        self.cpu_rng_state = torch.get_rng_state()
        self.cuda_rng_state = torch.cuda.get_rng_state()

    def _post_context_exec(self):
        """The callback function when exiting context.
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
                obj = self.layer_spec_dict[id(obj)]
            modified_args.append(obj)
        # (lyl)TODO: analyse kwargs as well
        modified_args = tuple(modified_args)
        self.root_id = module_id
        self.root_children = list(module.children())
        self.model = module
        layer_spec = LayerSpec(module.__class__, *modified_args, **kwargs)
        layer_spec.set_children(module.children())
        self.layer_spec_dict[module_id] = layer_spec
        for param in module.parameters(recurse=False):
            param.data = torch.rand(1, 1)

    def to_layer_list(self, exec_seq=None):

        if exec_seq is None:
            #if user do not provide the model executing sequence, we use the initialization order as the executing order.
            for child in self.root_children:
                layer_spec = self.layer_spec_dict[id(child)]
                if layer_spec.typename in (torch.nn.modules.container.ModuleList,
                                           torch.nn.modules.container.Sequential):
                    for child_in_container in layer_spec.children:
                        self.layer_spec_list.append(self.layer_spec_dict[id(child_in_container)])

                else:
                    self.layer_spec_list.append(layer_spec)

        else:
            func_key = "first"
            for index, element in enumerate(exec_seq):
                if isinstance(element, str):
                    module = dict(self.model.named_modules())[element]
                    layer_spec = self.layer_spec_dict[id(module)]
                    func_key = layer_spec
                    self.layer_spec_list.append(layer_spec)
                else:
                    if func_key not in self.func_dict:
                        self.func_dict[func_key] = []
                    self.func_dict[func_key].append(element)

    def partition(self, num_chunks, pipeline_size, rank):
        if isinstance(self.policy, str):
            if self.policy == "uniform":
                parts = partition_uniform(len(self.layer_spec_list), pipeline_size, num_chunks)[rank]
            elif self.policy == "balanced":
                param_counts = []
                for layer_spec in self.layer_spec_list:
                    param_counts.append(layer_spec.count_params())
                parts = partition_balanced(param_counts, pipeline_size, num_chunks)[rank]
            else:
                raise ValueError("A string partition policy should be one of ['uniform', 'balanced'].")
        elif isinstance(self.policy, dict):
            parts = self.policy[rank]
        else:
            raise ValueError("A partition policy should be either a string or a dictionary.")

        layers_to_build = []
        for start, end in parts:
            layers_to_build += self.layer_spec_list[start:end]
        func_dict_in_partition = {}
        module_list_in_partition = []
        if rank == 0 and "first" in self.func_dict:
            func_dict_in_partition["first"] = self.func_dict["first"]
        for layer in layers_to_build:
            module = layer.build()
            module_list_in_partition.append(module)
            if layer in self.func_dict:
                func_dict_in_partition[id(module)] = self.func_dict[layer]
        module_list_in_partition = torch.nn.ModuleList(module_list_in_partition)
        pipeline_model = PipelinableModel(module_list_in_partition, func_dict_in_partition)

        return pipeline_model

    def load_policy(self, policy):
        self.policy = policy


class PipelinableModel(torch.nn.Module):

    def __init__(self, module_list, func_dict):
        super().__init__()
        self.module_list = module_list
        self.func_dict = func_dict

    def forward(self, input_tensor):
        if "first" in self.func_dict:
            funcs = self.func_dict["first"]
            if isinstance(funcs, list):
                for f in funcs:
                    input_tensor = f(input_tensor)
            else:
                input_tensor = funcs(input_tensor)

        for module in self.module_list:
            input_tensor = module(input_tensor)
            if id(module) in self.func_dict:
                funcs = self.func_dict[id(module)]
                if isinstance(funcs, list):
                    for f in funcs:
                        input_tensor = f(input_tensor)
                else:
                    input_tensor = funcs(input_tensor)

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
        return self.typename(*recovered_args, **self.module_kwargs)

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
