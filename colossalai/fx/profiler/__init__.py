from .._compatibility import is_compatible_with_meta

if is_compatible_with_meta():
    from .memory import calculate_fwd_in, calculate_fwd_out, calculate_fwd_tmp
    from .opcount import flop_mapping
    from .profiler import profile_function, profile_method, profile_module
    from .tensor import MetaTensor
else:
    from .experimental import meta_profiler_function, meta_profiler_module, profile_function, profile_method, profile_module, calculate_fwd_in, calculate_fwd_tmp, calculate_fwd_out

from .dataflow import GraphInfo
from .memory import activation_size, is_inplace, parameter_size
