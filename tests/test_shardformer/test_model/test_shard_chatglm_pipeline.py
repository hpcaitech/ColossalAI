import copy
import os

import pytest
import torch

import colossalai
from colossalai.cluster import ProcessGroupMesh
from colossalai.logging import disable_existing_loggers
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer.policies.chatglm import ChatGLMForConditionalGenerationPolicy, ChatGLMModelPolicy
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


@parameterize('enable_fused_normalization', [False])
@parameterize('enable_tensor_parallelism', [False])
@parameterize('use_lazy_init', [False])
def run_chatglm_test(enable_fused_normalization, enable_tensor_parallelism, use_lazy_init):
    DP_DIM, PP_DIM = 0, 1
    DP_SIZE, PP_SIZE = 2, 2
    pg_mesh = ProcessGroupMesh(DP_SIZE, PP_SIZE)
    stage_manager = PipelineStageManager(pg_mesh, PP_DIM)
    sub_model_zoo = model_zoo.get_sub_registry('transformers_chatglm')
    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        # create new model for test
        inputs = data_gen_fn()
        inputs = {k: v.cuda() for k, v in inputs.items()}
        input_ids = inputs['input_ids']
        hidden_size = 64
        batch_size, seq_len = input_ids.shape
        hidden_state_shape = (seq_len, batch_size, hidden_size)
        if name == "transformers_chatglm":
            _, sharded_model = build_pipeline_model(model_fn, stage_manager, enable_fused_normalization,
                                                    enable_tensor_parallelism, use_lazy_init, ChatGLMModelPolicy())
            if stage_manager.is_last_stage():
                hidden_states = torch.randn(*hidden_state_shape).cuda()
                inputs['input_ids'] = None
                inputs['hidden_states'] = hidden_states
            outputs = sharded_model(**inputs)
            if stage_manager.is_last_stage():
                assert outputs[0].shape == hidden_state_shape

            else:
                assert outputs['hidden_states'].shape == hidden_state_shape

        if name == "transformers_chatglm_for_conditional_generation":
            _, sharded_model = build_pipeline_model(model_fn, stage_manager, enable_fused_normalization,
                                                    enable_tensor_parallelism, use_lazy_init,
                                                    ChatGLMForConditionalGenerationPolicy())
            if stage_manager.is_last_stage():
                hidden_states = torch.randn(*hidden_state_shape).cuda()
                inputs['input_ids'] = None
                inputs['hidden_states'] = hidden_states
            outputs = sharded_model(**inputs)
            if stage_manager.is_last_stage():
                assert outputs[0].shape == (batch_size, seq_len, 65024)
            else:
                assert outputs['hidden_states'].shape == hidden_state_shape

    torch.cuda.empty_cache()


def check_chatglm(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_chatglm_test()


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_chatglm():
    spawn(check_chatglm, 4)


if __name__ == "__main__":
    test_chatglm()
