from torch.fx import GraphModule

from .passes import ShapeProp, graph_profile_pass, shape_prop_pass
from .passes.graph_profile import FlopProfiler


def register_flop_count_impl(func):
    def wrapper(impl):
        FlopProfiler._custom_flop_count_impl[func] = impl
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
