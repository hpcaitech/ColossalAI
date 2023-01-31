import torch
import torch.fx
from torch.fx import GraphModule

from siu.fx.passes import GraphProfile, ShapeProp, graph_profile_pass, shape_prop_pass


def register_profile_impl(func):

    def wrapper(impl):
        GraphProfile._custom_profile_impl[func] = impl
        return impl

    return wrapper


def register_shape_impl(func):

    def wrapper(impl):
        ShapeProp._custom_dispatch_func[func] = impl
        return impl

    return wrapper


def symbolic_profile(module: GraphModule, *args, verbose=False) -> GraphModule:
    """Symbolically profile a model with sample inputs.

    Args:
        module (GraphModule): The module to be profiled
        args (Tuple): The sample inputs
        verbose (bool): Whether to print the profiling result

    Returns:
        GraphModule: The profiled module
    """
    module = shape_prop_pass(module, *args)
    module = graph_profile_pass(module, *args, verbose=verbose)
    return module
