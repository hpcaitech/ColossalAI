import pytest
import torch

from colossalai.kernel.kernel_loader import InferenceOpsLoader
from colossalai.utils import get_current_device

inference_ops = InferenceOpsLoader().load()


@pytest.mark.parametrize("SHAPE_X", [2])
@pytest.mark.parametrize("SHAPE_Y", [64])
@pytest.mark.parametrize("SHAPE_Z", [11008])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_silu_and_mul(SHAPE_X, SHAPE_Y, SHAPE_Z, dtype):
    torch.manual_seed(5)
    device = get_current_device()
    ref_input = torch.randn(SHAPE_X, SHAPE_Y, SHAPE_Z, dtype=dtype, device=device)
    origin_input = ref_input.clone()

    act_out = torch.nn.functional.silu(ref_input[0], inplace=True)
    ref_out = act_out * ref_input[1]

    origin_out = inference_ops.silu_and_mul(origin_input)

    if dtype == torch.float32:
        assert torch.allclose(origin_out, ref_out, atol=1e-5, rtol=1e-5)
    else:
        assert torch.allclose(origin_out, ref_out, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    test_silu_and_mul(2, 64, 11008, torch.float32)
    test_silu_and_mul(2, 64, 11008, torch.float16)
