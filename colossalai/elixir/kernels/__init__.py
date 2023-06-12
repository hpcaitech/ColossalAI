import torch.nn.functional as F

fused_torch_functions = {F.layer_norm: F.layer_norm}


def register_fused_layer_norm():
    try:
        from .layernorm import ln_func
        fused_torch_functions[F.layer_norm] = ln_func
        print('Register fused layer norm successfully from apex.')
    except:
        print('Cannot import fused layer norm, please install apex from source.')
        pass
