import pytest
import torch
import torch.distributed as dist

import colossalai
from colossalai.cluster import ProcessGroupMesh
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer.policies.auto_policy import get_autopolicy
from colossalai.shardformer.policies.base_policy import Policy
from colossalai.shardformer.shard import ShardConfig
from colossalai.testing import rerun_if_address_is_in_use, spawn
from tests.kit.model_zoo import model_zoo


def check_llama_model_policy():
    # create a LlamaModel
    PP_DIM = 0
    PP_SIZE = 2
    pg_mesh = ProcessGroupMesh(PP_SIZE)
    # print(pg_mesh)
    stage_manager = PipelineStageManager(pg_mesh, PP_DIM)
    sub_model_zoo = model_zoo.get_sub_registry('transformers_llama')
    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        model = model_fn()
        stage_manager = PipelineStageManager(pg_mesh, PP_DIM)
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


def run_dist_policy(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, port=port, host='localhost')
    check_llama_model_policy()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_llama_model_policy():
    spawn(run_dist_policy, 2)


if __name__ == "__main__":
    """test the llama model policy"""
    test_llama_model_policy()
