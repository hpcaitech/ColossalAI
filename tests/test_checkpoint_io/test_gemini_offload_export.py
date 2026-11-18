"""A CPU-offloaded Gemini export must contain the just-updated master values."""

import pytest
import torch
import torch.distributed as dist

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin
from colossalai.checkpoint_io import GeneralCheckpointIO
from colossalai.nn.optimizer import HybridAdam
from colossalai.testing import rerun_if_address_is_in_use, spawn


class RepeatedLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 8)
        self.head = torch.nn.Linear(8, 2)

    def forward(self, x):
        for _ in range(4):
            x = x + self.proj(x).tanh()
        return self.head(x)


def _run(rank, world_size, port, accumulation, shard, directory):
    torch.set_num_threads(1)
    colossalai.launch(rank=rank, world_size=world_size, port=port, host="127.0.0.1", seed=42)
    try:
        native = RepeatedLayer()
        names = [name for name, _ in native.named_parameters()]
        optimizer = HybridAdam(native.parameters(), lr=1e-2)
        booster = Booster(
            plugin=GeminiPlugin(
                precision="bf16",
                chunk_config_dict={world_size: {"chunk_size": 64, "keep_gathered": False}},
                placement_policy="static",
                offload_optim_frac=1,
                offload_param_frac=1,
                enable_gradient_accumulation=accumulation,
            )
        )
        model, optimizer, *_ = booster.boost(native, optimizer)
        before = model.state_dict(only_rank_0=False)
        loss = model(torch.randn(2, 8, device="cuda")).float().square().mean()
        booster.backward(loss, optimizer)
        optimizer.step()
        chunks = model.chunk_manager.get_chunks(model.fp16_params)
        assert all(not chunk.optim_sync_flag and chunk.device_type == "cpu" for chunk in chunks)
        expected = {
            name: optimizer.collect_states(i, only_rank_0=False)["_colossalai_master_param"].bfloat16().cpu().clone()
            for i, name in enumerate(names)
        }
        assert any(not torch.equal(expected[n], before[n]) for n in expected), "Optimizer did not update weights"
        # No intervening model forward is allowed before either export path.
        exported = model.state_dict(only_rank_0=False)
        for name, value in expected.items():
            torch.testing.assert_close(exported[name], value, rtol=0, atol=0, msg=name)
        assert all(not chunk.optim_sync_flag and chunk.device_type == "cpu" for chunk in chunks)
        path = str(directory) + ("/sharded" if shard else "/model.pt")
        booster.save_model(model, path, shard=shard)
        dist.barrier()
        loaded = RepeatedLayer().bfloat16()
        GeneralCheckpointIO().load_model(loaded, path, strict=True)
        for name, param in loaded.named_parameters():
            torch.testing.assert_close(param, expected[name], rtol=0, atol=0, msg=name)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("accumulation", [False, True])
@pytest.mark.parametrize("shard", [False, True])
@rerun_if_address_is_in_use()
def test_gemini_offload_export(accumulation, shard, tmp_path):
    assert torch.cuda.device_count() >= 2
    spawn(_run, 2, accumulation=accumulation, shard=shard, directory=str(tmp_path))
