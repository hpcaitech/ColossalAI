from typing import Dict
from colossalai.tensor import ColoParameter, ParallelAction, TensorSpec
from .modules import ColoModule
import torch

_COLOSSAL_MODULES: Dict[type, ColoModule] = {}


def register_colo_module(module_type: type, colo_module: ColoModule):
    global _COLOSSAL_MODULES
    _COLOSSAL_MODULES[module_type] = colo_module

def is_colo_module(module: torch.nn.Module):
    global _COLOSSAL_MODULES
    return type(module) in _COLOSSAL_MODULES

def get_colo_module(module: torch.nn.Module):
    global _COLOSSAL_MODULES
    if is_colo_module(module):
        colo_module = _COLOSSAL_MODULES[type(module)]
        colo_module.register()
        return colo_module
    else:
        return None

def check_colo_module(module: torch.nn.Module, recursive=True):
    if is_colo_module(module):
        colo_module = get_colo_module(module)
        param_names = colo_module.get_param_names()
        compute_pattern = None
        for param_name in param_names:
            param = module.get_parameter(param_name)
            if not isinstance(param, ColoParameter):
                raise Exception(f'Invalid ColoParameter spec: {param} in {module} is not a ColoParameter.')
            if param.has_spec():  
                cur_compute_pattern = param.spec.parallel_action.compute_pattern
                if compute_pattern is None:
                    compute_pattern = cur_compute_pattern
                else:
                    if cur_compute_pattern != compute_pattern:
                        raise Exception(f'Invalid ColoParameter spec: Params in {module} have different compute_pattern.')
            else:
                continue
            
        if compute_pattern is not None:
            if not colo_module.has_compute_pattern(compute_pattern):
                raise Exception(f'Invalid ColoParameter spec: ComputePattern {compute_pattern} in {module} is not allowed.')

            match_specs = False
            allowed_specs = colo_module.get_dist_specs(compute_pattern)
            for _, param_specs in allowed_specs.items():
                cur_match = True
                for param_name, dist_spec in param_specs.items():
                    param = module.get_parameter(param_name)
                    if param.has_spec():
                        if dist_spec != param.spec.dist_spec:
                            cur_match = False
                            break
                    else:
                        if dist_spec is not None:
                            cur_match = False
                            break
                if cur_match == True:
                    match_specs = True
                    break
            if match_specs == False:
                raise Exception(f'Invalid ColoParameter spec: Params in {module} are incorrectly sharded.')
    
    if recursive == True:
        for submodule in module.children():
            check_colo_module(submodule, recursive=True)

def init_colo_module(module: torch.nn.Module, parallel_action: ParallelAction, recursive=True, label='default'):
    compute_pattern = parallel_action.compute_pattern
    if is_colo_module(module):
        # for each param
        # set DistSpec and ParallelAction
        colo_module = get_colo_module(module)
        if not colo_module.has_compute_pattern_with_label(compute_pattern, label=label):
            raise NotImplementedError
        for param_name, dist_spec in colo_module.get_dist_specs_with_label(compute_pattern, label=label).items():
            if dist_spec is None:
                continue
            param = module.get_parameter(param_name)
            if isinstance(param, ColoParameter):
                spec = TensorSpec(dist_spec, parallel_action)
                param.set_spec(spec)
        check_colo_module(module, recursive=False)
    if recursive == True:
        for submodule in module.children():
            init_colo_module(submodule, parallel_action, recursive=True, label=label)
    