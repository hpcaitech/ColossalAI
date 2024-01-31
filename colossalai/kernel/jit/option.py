import torch

from colossalai.accelerator import get_accelerator
from colossalai.legacy.nn.layer.colossalai_layer import Embedding, Linear

from .bias_dropout_add import bias_dropout_add_fused_train
from .bias_gelu import bias_gelu_impl

JIT_OPTIONS_SET = False


def set_jit_fusion_options():
    """Set PyTorch JIT layer fusion options."""
    # LSG: the latest pytorch and CUDA versions may not support
    # the following jit settings
    global JIT_OPTIONS_SET
    if JIT_OPTIONS_SET == False:
        # flags required to enable jit fusion kernels
        TORCH_MAJOR = int(torch.__version__.split(".")[0])
        TORCH_MINOR = int(torch.__version__.split(".")[1])
        if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10):
            # nvfuser
            torch._C._jit_set_profiling_executor(True)
            torch._C._jit_set_profiling_mode(True)
            torch._C._jit_override_can_fuse_on_cpu(False)
            torch._C._jit_override_can_fuse_on_gpu(False)
            torch._C._jit_set_texpr_fuser_enabled(False)
            torch._C._jit_set_nvfuser_enabled(True)
            torch._C._debug_set_autodiff_subgraph_inlining(False)
        else:
            # legacy pytorch fuser
            torch._C._jit_set_profiling_mode(False)
            torch._C._jit_set_profiling_executor(False)
            torch._C._jit_override_can_fuse_on_cpu(True)
            torch._C._jit_override_can_fuse_on_gpu(True)

        JIT_OPTIONS_SET = True


def warmup_jit_fusion(
    batch_size: int,
    hidden_size: int,
    seq_length: int = 512,
    vocab_size: int = 32768,
    dtype: torch.dtype = torch.float32,
):
    """Compile JIT functions before the main training steps"""

    embed = Embedding(vocab_size, hidden_size).to(get_accelerator().get_current_device())
    linear_1 = Linear(hidden_size, hidden_size * 4, skip_bias_add=True).to(get_accelerator().get_current_device())
    linear_2 = Linear(hidden_size * 4, hidden_size, skip_bias_add=True).to(get_accelerator().get_current_device())

    x = torch.randint(
        vocab_size, (batch_size, seq_length), dtype=torch.long, device=get_accelerator().get_current_device()
    )
    x = embed(x)
    y, y_bias = linear_1(x)
    z, z_bias = linear_2(y)
    # Warmup JIT fusions with the input grad_enable state of both forward
    # prop and recomputation
    for bias_grad, input_grad in zip([True, True], [False, True]):
        for _ in range(10):
            bias = torch.rand_like(y_bias, dtype=dtype, device=get_accelerator().get_current_device())
            input_ = torch.rand_like(y, dtype=dtype, device=get_accelerator().get_current_device())
            bias.requires_grad, input_.requires_grad = bias_grad, input_grad
            bias_gelu_impl(input_, bias)

    # Warmup fused bias+dropout+add
    dropout_rate = 0.1
    # Warmup JIT fusions with the input grad_enable state of both forward
    # prop and recomputation
    for input_grad, bias_grad, residual_grad in zip([False, True], [True, True], [True, True]):
        for _ in range(10):
            input_ = torch.rand_like(z, dtype=dtype, device=get_accelerator().get_current_device())
            residual = torch.rand_like(x, dtype=dtype, device=get_accelerator().get_current_device())
            bias = torch.rand_like(z_bias, dtype=dtype, device=get_accelerator().get_current_device())
            input_.requires_grad = input_grad
            bias.requires_grad = bias_grad
            residual.requires_grad = residual_grad
            bias_dropout_add_fused_train(input_, bias, residual, dropout_rate)

    torch.cuda.empty_cache()
