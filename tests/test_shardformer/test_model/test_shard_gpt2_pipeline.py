import pytest
import torch

import colossalai
from colossalai.cluster import ProcessGroupMesh
from colossalai.logging import disable_existing_loggers
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.testing import (
    assert_hf_output_close,
    clear_cache_before_run,
    parameterize,
    rerun_if_address_is_in_use,
    spawn,
)
from tests.kit.model_zoo import model_zoo
from tests.test_shardformer.test_model._utils import build_model, build_pipeline_model, run_forward


def check_forward_backward(org_model, sharded_model, data_gen_fn, output_transform_fn, loss_fn):
    # TODO: add tests for forward/backward later
    pass


@parameterize('enable_fused_normalization', [False])
@parameterize('enable_tensor_parallelism', [False])
@parameterize('use_lazy_init', [False])
#TODO: merge this into test_shard_gpt2
def run_gpt2_test(enable_fused_normalization, enable_tensor_parallelism, use_lazy_init):
    DP_DIM, PP_DIM = 0, 1
    DP_SIZE, PP_SIZE = 2, 2
    pg_mesh = ProcessGroupMesh(DP_SIZE, PP_SIZE)
    stage_manager = PipelineStageManager(pg_mesh, PP_DIM)

    sub_model_zoo = model_zoo.get_sub_registry('transformers_gpt')
    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        if name != "transformers_gpt":
            continue

        inputs = data_gen_fn()
        inputs = {k: v.cuda() for k, v in inputs.items()}

        org_model, sharded_model = build_pipeline_model(model_fn, stage_manager, enable_fused_normalization,
                                                        enable_tensor_parallelism, use_lazy_init)
        org_model.train()
        org_output = org_model(**inputs)
        hidden_state_shape = org_output['last_hidden_state'].shape

        if stage_manager.is_first_stage():
            output = sharded_model(**inputs)
            assert output['hidden_states'].shape == hidden_state_shape
        else:
            attention_mask = inputs['attention_mask']
            hidden_states = torch.zeros(*hidden_state_shape).cuda()
            output = sharded_model(hidden_states=hidden_states, attention_mask=attention_mask)
            if stage_manager.is_last_stage():
                assert output['last_hidden_state'].shape == hidden_state_shape
            else:
                assert output['hidden_states'].shape == hidden_state_shape

    torch.cuda.empty_cache()


def check_gpt2(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_gpt2_test()


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_gpt2():
    spawn(check_gpt2, 4)


if __name__ == "__main__":
    test_gpt2()
