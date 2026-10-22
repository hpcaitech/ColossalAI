"""Qwen loss-wrapper checks with a real, single-rank CPU Gloo process group.

This exercises dist_cross_entropy's non-TP path, not vocabulary-parallel CE.
"""

from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import transformers
from packaging.version import Version

if Version(transformers.__version__) < Version("5.17.0"):
    pytest.skip("HF 5.17 contract", allow_module_level=True)

from test_qwen import close, inputs, local_module, make_pair, patch_model

from colossalai.shardformer.shard import ShardConfig

pytestmark = pytest.mark.skipif(Version(transformers.__version__) < Version("5.17.0"), reason="HF 5.17 contract")


@pytest.fixture(scope="module")
def single_rank_gloo(tmp_path_factory):
    if dist.is_initialized():
        pytest.skip("This isolated CPU test must own its single-rank process group")
    rendezvous = tmp_path_factory.mktemp("qwen-gloo") / "rendezvous"
    dist.init_process_group("gloo", init_method=rendezvous.as_uri(), rank=0, world_size=1)
    yield
    dist.destroy_process_group()


@pytest.mark.parametrize("pipeline,dtype", [(False, torch.float32), (True, torch.float32), (True, torch.bfloat16)])
def test_lm_loss_wrapper(single_rank_gloo, family, pipeline, dtype):
    reference, actual = make_pair(family)
    reference, actual = reference.to(dtype), actual.to(dtype)
    module = local_module(family)
    config = ShardConfig(enable_tensor_parallelism=False)
    if pipeline:
        forward = getattr(getattr(module, "Qwen" + family[-1] + "PipelineForwards"), f"{family}_for_causal_lm_forward")
        manager = SimpleNamespace(is_first_stage=lambda: True, is_last_stage=lambda: True)
        kwargs = dict(stage_manager=manager, stage_index=[0, 2], shard_config=config)
    else:
        patch_model(actual.model, family)
        forward = module.get_lm_forward_with_dist_cross_entropy(config)
        kwargs = {}
    actual.forward = MethodType(forward, actual)
    ref_out, out = reference(**inputs("left")), actual(**inputs("left"), **kwargs)
    assert out.loss.dtype == ref_out.loss.dtype == torch.float32
    close(out.logits, ref_out.logits)
    close(out.loss, ref_out.loss)
    ref_out.loss.backward()
    out.loss.backward()
    for a, b in zip(actual.parameters(), reference.parameters()):
        assert a.grad is not None and b.grad is not None
        close(a.grad, b.grad, atol=1e-6, rtol=1e-4)
