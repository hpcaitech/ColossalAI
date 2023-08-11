import torch
from torch import nn
import triton
import triton.language as tl
from transformers.models.llama.modeling_llama import LlamaRMSNorm

@triton.jit
def rms_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = x * rstd
        y = x_hat * w
        # Write output
        tl.store(Y + cols, y, mask=mask)

class TritonLlamaRMSNorm(nn.Module):
    def __init__(self, weight, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = weight
        self.variance_epsilon = eps

    def forward(self, x):
        with torch.cuda.device(x.device):
            y = torch.empty_like(x)
            # reshape input data into 2D tensor
            x_arg = x.reshape(-1, x.shape[-1])
            M, N = x_arg.shape
            # Less than 64KB per feature: enqueue fused kernel
            MAX_FUSED_SIZE = 65536 // x.element_size()
            BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
            if N > BLOCK_SIZE:
                raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
            # heuristics for number of warps
            num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
            # enqueue kernel
            rms_norm_fwd_fused[(M,)](x_arg, y, self.weight, 
                                    x_arg.stride(0), N, self.variance_epsilon,
                                    BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
        return y
        
        
def make_quant_norm(model):
    """
    Replace all LlamaRMSNorm modules with TritonLlamaRMSNorm modules
    """

    for name, m in model.named_modules():
        if not isinstance(m, LlamaRMSNorm):
            continue

        norm = TritonLlamaRMSNorm(m.weight, m.variance_epsilon)

        if '.' in name:
            parent_name = name.rsplit('.', 1)[0]
            child_name = name[len(parent_name) + 1:]
            parent = model.get_submodule(parent_name)
        else:
            parent_name = ''
            parent = model
            child_name = name

        #print(f"Replacing {name} with quant_attn; parent: {parent_name}, child's name: {child_name}")

        setattr(parent, child_name, norm)
