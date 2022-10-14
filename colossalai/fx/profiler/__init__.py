from ... import META_COMPATIBILITY
if META_COMPATIBILITY:
    from .opcount import flop_mapping
    from .tensor import MetaTensor
    from .profiler import profile_function, profile_method, profile_module
    from .memory import calculate_fwd_in, calculate_fwd_tmp, calculate_fwd_out
else:
    from .experimental import meta_profiler_function, meta_profiler_module, profile_function, profile_method, profile_module, calculate_fwd_in, calculate_fwd_tmp, calculate_fwd_out

from .dataflow import GraphInfo
from .memory import parameter_size, activation_size, is_inplace
