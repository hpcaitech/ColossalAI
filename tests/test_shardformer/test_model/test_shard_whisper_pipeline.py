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


def check_whisper_model_policy(name, model: torch.nn.Module, stage_manager: PipelineStageManager):
    policy = get_autopolicy(model)
    policy.set_model(model)
    model_config = ShardConfig(pipeline_stage_manager=stage_manager, enable_tensor_parallelism=False)
    policy.set_shard_config(model_config)
    layers = policy.get_held_layers()
    if stage_manager.is_first_stage():
        assert len(layers) == 2 + 2
    else:
        if name == "transformers_whisper":
            assert len(layers) == 2 + 3
        if name == "transformers_whisper_for_audio_classification":
            assert len(layers) == 2 + 2
        else:
            assert len(layers) == 2 + 4


def check_whisper_model_pipeline_forward(name, sharded_model, stage_manager: PipelineStageManager):
    x = torch.rand(1, 80, 3000).cuda()
    if stage_manager.stage == 0:
        #attention_mask = torch.ones_like(x).cuda()
        output = sharded_model(input_features=x,
        #attention_mask=attention_mask
                              )
        print(output)
    else:
        hidden_of_audio = torch.rand(1, 1500, 256).to(torch.float32).cuda()
        #hidden_states = torch.rand(1, 80, 3000).to(torch.float32).cuda()
        #attention_mask = torch.ones((80, 3000)).cuda()
        encoder_output_states = torch.zeros(*(2, 3, 256)).cuda()
        encoder_outputs = (encoder_output_states,)
        #decoder_input_ids = torch.tensor([[1, 1]]) * 50258
        output = sharded_model(
        #decoder_input_ids=decoder_input_ids.cuda(),
            hidden_states=hidden_of_audio,
        #attention_mask=attention_mask,
            encoder_outputs=encoder_outputs)
        print('final output', output[0])
        assert output[0] is not None


@parameterize('enable_fused_normalization', [False])
@parameterize('enable_tensor_parallelism', [False])
@parameterize('use_lazy_init', [False])
#TODO: merge this into test_shard_whisper
def run_whisper_test(enable_fused_normalization, enable_tensor_parallelism, use_lazy_init):
    PP_DIM = 0
    PP_SIZE = 2
    pg_mesh = ProcessGroupMesh(PP_SIZE)
    stage_manager = PipelineStageManager(pg_mesh, PP_DIM)

    sub_model_zoo = model_zoo.get_sub_registry('transformers_whisper')

    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        if name != "transformers_whisper_for_audio_classification":
            continue
        org_model, sharded_model = build_pipeline_model(model_fn, stage_manager, enable_fused_normalization,
                                                        enable_tensor_parallelism, use_lazy_init)
        check_whisper_model_policy(name, org_model, stage_manager)
        check_whisper_model_pipeline_forward(name, sharded_model, stage_manager)

    torch.cuda.empty_cache()


def check_whisper(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_whisper_test()


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_whisper():
    spawn(check_whisper, 2)


if __name__ == "__main__":
    test_whisper()
