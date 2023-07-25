import pytest
import torch

import colossalai
from colossalai.cluster import ProcessGroupMesh
from colossalai.logging import disable_existing_loggers
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer.policies.t5 import T5BasePolicy
from colossalai.testing import clear_cache_before_run, parameterize, rerun_if_address_is_in_use, spawn
from tests.kit.model_zoo import model_zoo
from tests.test_shardformer.test_model._utils import build_pipeline_model


def check_forward_backward(org_model, sharded_model, data_gen_fn, output_transform_fn, loss_fn):
    # TODO: add tests for forward/backward later
    pass


@parameterize('enable_tensor_parallelism', [False])
@parameterize('enable_fused_normalization', [False])
@parameterize('use_lazy_init', [False])
#TODO: merge this into test_shard_t5.py
def run_t5_test(enable_fused_normalization, enable_tensor_parallelism, use_lazy_init):
    DP_DIM, PP_DIM = 0, 1
    DP_SIZE, PP_SIZE = 2, 2
    pg_mesh = ProcessGroupMesh(DP_SIZE, PP_SIZE)
    stage_manager = PipelineStageManager(pg_mesh, PP_DIM)

    sub_model_zoo = model_zoo.get_sub_registry('transformers_t5')
    for name, (model_fn, data_gen_fn, _, _, _) in sub_model_zoo.items():

        inputs = data_gen_fn()
        inputs = {k: v.cuda() for k, v in inputs.items()}
        input_ids = inputs['input_ids']

        _, sharded_model = build_pipeline_model(model_fn, stage_manager, enable_fused_normalization,
                                                enable_tensor_parallelism, use_lazy_init)

        batch_size, seq_len = input_ids.shape
        hidden_size = sharded_model.config.d_model
        num_heads = sharded_model.config.num_heads
        hidden_state_shape = (batch_size, seq_len, hidden_size)
        position_bias_shape = (batch_size, num_heads, seq_len, seq_len)

        num_encoder_layers = len(sharded_model.encoder.block)
        decoder = sharded_model.__dict__.get('decoder', None)
        num_decoder_layers = len(decoder.block) if decoder else 0

        _, decoder_starting_stage = T5BasePolicy.distribute_t5_layers(num_encoder_layers, num_decoder_layers, PP_SIZE)
        stage = stage_manager.stage
        at_first_stage = (stage == 0) or (stage == decoder_starting_stage)
        at_last_stage = (stage == decoder_starting_stage - 1) or (stage == stage_manager.num_stages - 1)
        in_decoder = stage >= decoder_starting_stage

        if not at_first_stage:
            # change inputs if not the first stage
            hidden_states = torch.zeros(*hidden_state_shape).cuda()
            position_bias = torch.zeros(*position_bias_shape).cuda()
            encoder_decoder_position_bias = torch.zeros(*position_bias_shape).cuda()
            inputs['input_ids'] = None
            inputs['hidden_states'] = hidden_states
            inputs['position_bias'] = position_bias
            inputs['encoder_decoder_position_bias'] = encoder_decoder_position_bias
        if in_decoder:
            encoder_output_states = torch.zeros(*hidden_state_shape).cuda()
            inputs['encoder_outputs'] = (encoder_output_states,)

        sharded_model.train()
        output = sharded_model(**inputs)
        if at_last_stage:
            if name == 'transformers_t5_for_conditional_generation' and in_decoder:
                assert output.loss is not None
            else:
                if name != 'transformers_t5_encoder_model' and not in_decoder:
                    output = output['encoder_outputs']
                assert output[0].shape == hidden_state_shape
        else:
            assert output['hidden_states'].shape == hidden_state_shape
            # position_bias information should be passed in T5
            assert output['position_bias'].shape == position_bias_shape
            if in_decoder:
                assert output['encoder_decoder_position_bias'].shape == position_bias_shape

    torch.cuda.empty_cache()


def check_t5(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_t5_test()


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_t5():
    spawn(check_t5, 4)


if __name__ == "__main__":
    test_t5()
