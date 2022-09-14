from ... import META_COMPATIBILITY
if META_COMPATIBILITY:
    from .opcount import flop_mapping
    from .tensor import MetaTensor
    from .profiler import profile_function, profile_method, profile_module
else:
    from .experimental import meta_profiler_function, meta_profiler_module, profile_function, profile_method, profile_module

from .dataflow import GraphInfo
from .memory import parameter_size, activation_size
