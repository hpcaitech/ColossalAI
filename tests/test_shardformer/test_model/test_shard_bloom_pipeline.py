import pytest
import torch

import colossalai
from colossalai.cluster import ProcessGroupMesh
from colossalai.logging import disable_existing_loggers
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer.policies.auto_policy import get_autopolicy
from colossalai.shardformer.policies.base_policy import Policy
from colossalai.shardformer.shard import ShardConfig
from colossalai.tensor.d_tensor.api import is_customized_distributed_tensor, is_distributed_tensor
from colossalai.testing import (
    assert_hf_output_close,
    clear_cache_before_run,
    parameterize,
    rerun_if_address_is_in_use,
    spawn,
)
from tests.kit.model_zoo import model_zoo
from tests.test_shardformer.test_model._utils import build_model, build_pipeline_model, run_forward


def check_bloom_model_policy(name, model: torch.nn.Module, stage_manager: PipelineStageManager):
    policy = get_autopolicy(model)
    policy.set_model(model)
    model_config = ShardConfig(pipeline_stage_manager=stage_manager, enable_tensor_parallelism=False)
    policy.set_shard_config(model_config)
    layers = policy.get_held_layers()
    if stage_manager.is_first_stage():
        assert len(layers) == 0 + 2
    else:
        if name == 'transformers_bloom':
            assert len(layers) == 1 + 1
        elif name == 'transformers_bloom_for_token_classification':
            assert len(layers) == 1 + 3
        else:
            assert len(layers) == 1 + 2


def check_bloom_model_pipeline_forward(name, sharded_model, stage_manager: PipelineStageManager):
    if stage_manager.stage == 0:
        x = torch.randint(0, 1000, (1, 3)).cuda()
        attention_mask = torch.ones_like(x).cuda()
        output = sharded_model(input_ids=x, attention_mask=attention_mask)
        assert output['hidden_states'].shape == (1, 3, 64)
    else:
        attention_mask = torch.ones((1, 3)).cuda()
        hidden_states = torch.randint(0, 1000, (1, 3, 64)).to(torch.float32).cuda()
        output = sharded_model(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        assert output[0].shape[0] == 1


@parameterize('enable_fused_normalization', [False])
@parameterize('enable_tensor_parallelism', [False])
@parameterize('use_lazy_init', [False])
#TODO: merge this into test_shard_bloom
def run_bloom_test(enable_fused_normalization, enable_tensor_parallelism, use_lazy_init):
    PP_DIM = 0
    PP_SIZE = 2
    pg_mesh = ProcessGroupMesh(PP_SIZE)
    stage_manager = PipelineStageManager(pg_mesh, PP_DIM)

    sub_model_zoo = model_zoo.get_sub_registry('transformers_bloom')
    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        org_model, sharded_model = build_pipeline_model(model_fn, stage_manager, enable_fused_normalization,
                                                        enable_tensor_parallelism, use_lazy_init)
        check_bloom_model_policy(name, org_model, stage_manager)
        check_bloom_model_pipeline_forward(name, sharded_model, stage_manager)

    torch.cuda.empty_cache()


def check_bloom(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_bloom_test()


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_bloom():
    spawn(check_bloom, 2)


if __name__ == "__main__":
    test_bloom()
