try:
    import torch_int

    HAS_TORCH_INT = True
except ImportError:
    HAS_TORCH_INT = False
    raise ImportError(
        "Not install torch_int. Please install torch_int from https://github.com/Guangxuan-Xiao/torch-int"
    )

if HAS_TORCH_INT:
    from .llama import LLamaSmoothquantAttention, LlamaSmoothquantMLP
