import warnings

import pytest
import torch
from packaging import version

try:
    from colossalai.kernel.op_builder.smoothquant import SmoothquantBuilder

    smoothquant_cuda = SmoothquantBuilder().load()
    HAS_SMOOTHQUANT_CUDA = True
except:
    warnings.warn("CUDA smoothquant linear is not installed")
    HAS_SMOOTHQUANT_CUDA = False


try:
    from colossalai.inference.quant.smoothquant.models import LlamaSmoothquantMLP

    HAS_TORCH_INT = True
except:
    HAS_TORCH_INT = False
    warnings.warn("Please install torch_int from https://github.com/Guangxuan-Xiao/torch-int")


CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse("11.4")


def torch_llama_mlp(gate_proj, up_proj, down_proj, x):
    gate_out = torch.mm(x, gate_proj)
    silu = torch.nn.SiLU()
    gate_out = silu(gate_out)
    up_out = torch.mm(x, up_proj)

    o_out = gate_out * up_out

    max_up = torch.max(torch.abs(o_out))
    min_up = torch.min(torch.abs(o_out))

    torch_out = torch.mm(o_out, down_proj)

    return (torch_out, max_up, min_up)


@pytest.mark.skipif(
    not CUDA_SUPPORT or not HAS_SMOOTHQUANT_CUDA or not HAS_TORCH_INT,
    reason="smoothquant linear not installed properly or not install torch_int",
)
def test_llama_mlp():
    hidden_size = 256
    intermediate_size = 512

    smooth_mlp = LlamaSmoothquantMLP(intermediate_size, hidden_size)

    smooth_mlp.gate_proj.weight = torch.ones((intermediate_size, hidden_size), dtype=torch.int8, device="cuda")

    smooth_mlp.up_proj.weight = torch.randint(
        -10, 10, (intermediate_size, hidden_size), dtype=torch.int8, device="cuda"
    )
    smooth_mlp.down_proj.weight = torch.randint(
        -10, 10, (hidden_size, intermediate_size), dtype=torch.int8, device="cuda"
    )

    x = torch.ones((1, 256), dtype=torch.int8, device="cuda")

    torch_out, max_inter, min_inter = torch_llama_mlp(
        smooth_mlp.gate_proj.weight.transpose(0, 1).to(torch.float) / hidden_size,
        smooth_mlp.up_proj.weight.transpose(0, 1).to(torch.float) / 127,
        smooth_mlp.down_proj.weight.transpose(0, 1).to(torch.float) / 127,
        x.to(torch.float),
    )

    smooth_mlp.down_proj_input_scale = torch.tensor(max_inter.item() / 127)
    smooth_mlp.gate_proj.a = torch.tensor(1 / hidden_size)
    smooth_mlp.up_proj.a = torch.tensor(1 / 127)
    smooth_mlp.down_proj.a = torch.tensor(1 / 127 * (max_inter.item() / 127))

    smooth_out = smooth_mlp(x)

    assert torch.allclose(torch_out, smooth_out, rtol=1e-02, atol=1e-01)


if __name__ == "__main__":
    test_llama_mlp()
