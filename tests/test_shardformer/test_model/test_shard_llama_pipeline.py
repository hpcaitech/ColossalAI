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


def check_llama_model_policy(name, model: torch.nn.Module, stage_manager: PipelineStageManager):
    policy = get_autopolicy(model)
    policy.set_model(model)
    model_config = ShardConfig(pipeline_stage_manager=stage_manager, enable_tensor_parallelism=False)
    policy.set_shard_config(model_config)
    layers = policy.get_held_layers()
    if stage_manager.is_first_stage():
        assert len(layers) == 2 + 1
    else:
        if name == "transformers_llama":
            assert len(layers) == 2 + 1
        else:
            assert len(layers) == 2 + 2


def check_llama_model_pipeline_forward(name, sharded_model, stage_manager: PipelineStageManager):
    x = torch.randint(0, 1000, (2, 3)).cuda()
    if stage_manager.stage == 0:
        attention_mask = torch.ones_like(x).cuda()
        output = sharded_model(input_ids=x, attention_mask=attention_mask)
        assert output['hidden_states'].shape == (2, 3, 128)
    else:
        hidden_states = torch.randint(0, 1000, (2, 3, 128)).to(torch.float32).cuda()
        attention_mask = torch.ones((2, 3)).cuda()
        output = sharded_model(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        assert output[0] is not None


@parameterize('enable_fused_normalization', [False])
@parameterize('enable_tensor_parallelism', [False])
@parameterize('use_lazy_init', [False])
#TODO: merge this into test_shard_llama
def run_llama_test(enable_fused_normalization, enable_tensor_parallelism, use_lazy_init):
    PP_DIM = 0
    PP_SIZE = 2
    pg_mesh = ProcessGroupMesh(PP_SIZE)
    stage_manager = PipelineStageManager(pg_mesh, PP_DIM)

    sub_model_zoo = model_zoo.get_sub_registry('transformers_llama')

    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        org_model, sharded_model = build_pipeline_model(model_fn, stage_manager, enable_fused_normalization,
                                                        enable_tensor_parallelism, use_lazy_init)
        check_llama_model_policy(name, org_model, stage_manager)
        check_llama_model_pipeline_forward(name, sharded_model, stage_manager)

    torch.cuda.empty_cache()


def check_llama(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_llama_test()


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_llama():
    spawn(check_llama, 2)


if __name__ == "__main__":
    test_llama()
