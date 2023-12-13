try:
    import triton

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("Triton is not installed. Please install Triton to use Triton kernels.")

# There may exist import error even if we have triton installed.
if HAS_TRITON:
    from .fused_layernorm import layer_norm
    from .gptq_triton import gptq_fused_linear_triton
    from .softmax import softmax

    __all__ = [
        "softmax",
        "layer_norm",
        "gptq_fused_linear_triton",
    ]
