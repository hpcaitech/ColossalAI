import pytest
import torch
from packaging.version import Version

from colossalai.checkpoint_io import GeneralCheckpointIO


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(Version(torch.__version__) < Version("2.10"), reason="loading .bin optimizer shards requires 2.10")
@pytest.mark.parametrize("mode", ["capturable", "fused"])
def test_sharded_adamw_step_resume(tmp_path, mode):
    # Each active parameter has enough optimizer state to fill its own shard.
    params = [torch.nn.Parameter(torch.ones(150_000, device="cuda")) for _ in range(3)]
    original = torch.optim.AdamW([{"params": params[:2], mode: True}, {"params": params[2:]}])
    for param in (params[0], params[2]):
        param.grad = torch.full_like(param, 0.25)
    original.step()
    io = GeneralCheckpointIO()
    io.save_optimizer(original, str(tmp_path), shard=True, size_per_shard=1)
    assert len(list(tmp_path.glob("pytorch_optim-*.bin"))) >= 2
    restored_params = [torch.nn.Parameter(p.detach().clone()) for p in params]
    # The checkpoint carries the flags, even if the destination uses defaults.
    restored = torch.optim.AdamW([{"params": restored_params[:2]}, {"params": restored_params[2:]}])
    io.load_optimizer(restored, str(tmp_path))
    assert restored_params[1] not in restored.state
    for i in (0, 2):
        torch.testing.assert_close(restored.state[restored_params[i]]["step"], original.state[params[i]]["step"])
        params[i].grad = torch.full_like(params[i], -0.125)
        restored_params[i].grad = params[i].grad.clone()
    original.step()
    restored.step()
    for i in (0, 2):
        torch.testing.assert_close(restored_params[i], params[i])
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(restored.state[restored_params[i]][key], original.state[params[i]][key])
