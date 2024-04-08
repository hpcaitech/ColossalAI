import pytest
import torch
from packaging import version

try:
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("please install triton from https://github.com/openai/triton")

try:
    from auto_gptq.modeling._utils import autogptq_post_init
    from auto_gptq.utils.import_utils import dynamically_import_QuantLinear
    from exllama_kernels import prepare_buffers, set_tuning_params

    from colossalai.inference.quant.gptq import CaiQuantLinear

    HAS_AUTO_GPTQ = True
except:
    HAS_AUTO_GPTQ = False
    print("please install AutoGPTQ from https://github.com/PanQiWei/AutoGPTQ")

import warnings

HAS_GPTQ_CUDA = False
try:
    from colossalai.kernel.op_builder.gptq import GPTQBuilder

    gptq_cuda = GPTQBuilder().load()
    HAS_GPTQ_CUDA = True
except ImportError:
    warnings.warn("CUDA gptq is not installed")
    HAS_GPTQ_CUDA = False

TRITON_CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse("11.4")

max_inner_outer_dim = 1
max_input_len = 1
max_dq_buffer_size = 1
gptq_temp_dq_buffer = None
gptq_temp_state_buffer = None


def init_buffer(cai_linear, use_act_order=False):
    global max_dq_buffer_size
    global max_input_len
    global max_dq_buffer_size
    global max_inner_outer_dim
    global gptq_temp_dq_buffer
    global gptq_temp_state_buffer

    max_dq_buffer_size = max(max_dq_buffer_size, cai_linear.qweight.numel() * 8)

    if use_act_order:
        max_inner_outer_dim = max(max_inner_outer_dim, cai_linear.infeatures, cai_linear.outfeatures)

    if use_act_order:
        max_input_len = 4096
    # The temp_state buffer is required to reorder X in the act-order case.
    # The temp_dq buffer is required to dequantize weights when using cuBLAS, typically for the prefill.
    gptq_temp_state_buffer = torch.zeros(
        (max_input_len, max_inner_outer_dim), dtype=torch.float16, device=torch.cuda.current_device()
    )
    gptq_temp_dq_buffer = torch.zeros((1, max_dq_buffer_size), dtype=torch.float16, device=torch.cuda.current_device())

    gptq_cuda.prepare_buffers(torch.device(torch.cuda.current_device()), gptq_temp_state_buffer, gptq_temp_dq_buffer)
    # Using the default from exllama repo here.
    matmul_recons_thd = 8
    matmul_fused_remap = False
    matmul_no_half2 = False
    gptq_cuda.set_tuning_params(matmul_recons_thd, matmul_fused_remap, matmul_no_half2)


@pytest.mark.skipif(
    not TRITON_CUDA_SUPPORT or not HAS_TRITON or not HAS_AUTO_GPTQ,
    reason="triton requires cuda version to be higher than 11.4 or not install auto-gptq",
)
def test_gptq_linear():
    infeature = 1024
    outfeature = 1024
    group_size = 128
    wbits = 4

    inps = torch.ones(1, 1, infeature).to(torch.float16).to(torch.cuda.current_device())
    batch_inps = torch.randn(1, 16, infeature).to(torch.float16).to(torch.cuda.current_device())

    device = torch.device("cuda:0")

    linear_class = dynamically_import_QuantLinear(use_triton=False, desc_act=False, group_size=group_size, bits=wbits)

    linear = linear_class(
        bits=4,
        group_size=group_size,
        infeatures=infeature,
        outfeatures=outfeature,
        bias=False,
    )

    torch.manual_seed(42)

    linear.qweight = torch.randint(-100, 100, size=linear.qweight.shape, dtype=torch.int32)
    linear.scales = linear.scales + 0.002

    linear = linear.to(device)

    cai_linear = CaiQuantLinear(wbits, group_size, infeature, outfeature, True)
    cai_linear.qweight.data.copy_(linear.qweight)
    cai_linear.scales = cai_linear.scales + 0.002
    cai_linear = cai_linear.to(device)

    linear = autogptq_post_init(linear, use_act_order=False)

    max_inner_outer_dim = max(infeature, outfeature)
    max_dq_buffer_size = linear.infeatures * linear.outfeatures
    max_input_len = 2048
    buffers = {
        "temp_state": torch.zeros((max_input_len, max_inner_outer_dim), dtype=torch.float16, device=device),
        "temp_dq": torch.zeros((1, max_dq_buffer_size), dtype=torch.float16, device=device),
    }

    prepare_buffers(device, buffers["temp_state"], buffers["temp_dq"])

    # Using the default from exllama repo here.
    matmul_recons_thd = 8
    matmul_fused_remap = False
    matmul_no_half2 = False
    set_tuning_params(matmul_recons_thd, matmul_fused_remap, matmul_no_half2)

    with torch.no_grad():
        gptq_out = linear(inps)
        batch_gptq_out = linear(batch_inps)
        torch.cuda.synchronize()
        cai_out = cai_linear(inps)
        torch.cuda.synchronize()

        batch_cai_out = cai_linear(batch_inps)
        torch.cuda.synchronize()

    assert torch.allclose(cai_out, gptq_out, rtol=1e-01, atol=1e-01)
    assert torch.allclose(batch_cai_out, batch_gptq_out, rtol=1e-01, atol=1e-01)


if __name__ == "__main__":
    test_gptq_linear()
