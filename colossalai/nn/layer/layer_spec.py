import torch
import functools
from colossalai.zero.init_ctx.init_context import _substitute_init_recursively, InsertPostInitMethodToModuleSubClasses


def call_to_str(base, *args, **kwargs):
    """Construct a string representation of a call.

    Args:
        base (str): name of the call
        args (tuple, optional): args to ``base``
        kwargs (dict, optional): kwargs supplied to ``base``

    Returns:
        str: A string representation of base(*args, **kwargs)
    """
    name = f'{base}('
    if args:
        name += ', '.join(repr(arg) for arg in args)
        if kwargs:
            name += ', '
    if kwargs:
        name += ', '.join(f'{key}={repr(arg)}' for key, arg in kwargs.items())
    name += ')'
    return name


class LayerSpecContext(InsertPostInitMethodToModuleSubClasses):

    def __init__(self):
        super().__init__()
        self.layer_spec_dict = {}
        self.root_id = None
        self.root_children = None

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
        self.root_id = module_id
        self.root_children = list(module.children())
        layer_spec = LayerSpec(module.__class__, *args, **kwargs)
        layer_spec.get_children(module.children())
        self.layer_spec_dict[module_id] = layer_spec
        for param in module.parameters(recurse=False):
            param.data = torch.zeros(1, 1)


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
        return self.typename(*self.module_args, **self.module_kwargs)

    def get_children(self, children):
        self.children = children

    def count_params(self):
        layer = self.build()
        for param in layer.parameters():
            self._param_count += param.numel()
        return self._param_count

    def reset_param_count(self):
        self._param_count = 0
