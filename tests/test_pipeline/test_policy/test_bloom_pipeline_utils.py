import pytest
import torch
import torch.distributed as dist
from transformers.models.bloom import BloomConfig, BloomModel

import colossalai
from colossalai.cluster import ProcessGroupMesh
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer.policies.base_policy import Policy
from colossalai.shardformer.policies.bloom import BloomModelPolicy
from colossalai.shardformer.shard import ShardConfig
from colossalai.testing import rerun_if_address_is_in_use, spawn
from tests.kit.model_zoo import model_zoo
from tests.test_shardformer.test_model._utils import build_pipeline_model


def check_bloom_model_policy():
    # create a BloomModel
    PP_DIM = 0
    PP_SIZE = 2
    pg_mesh = ProcessGroupMesh(PP_SIZE)
    # print(pg_mesh)
    stage_manager = PipelineStageManager(pg_mesh, PP_DIM)
    sub_model_zoo = model_zoo.get_sub_registry('transformers_bloom')
    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        model = model_fn()
        stage_manager = PipelineStageManager(pg_mesh, PP_DIM)
        policy = BloomModelPolicy()
        policy.set_model(model)
        model_config = ShardConfig(pipeline_stage_manager=stage_manager, enable_tensor_parallelism=False)
        policy.set_shard_config(model_config)
        layers = policy.get_held_layers()
        if stage_manager.is_first_stage():
            assert len(layers) == 0 + 2
        else:
            assert len(layers) == 1 + 1


def run_dist_policy(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, port=port, host='localhost')
    check_bloom_model_policy()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_bloom_model_policy():
    spawn(run_dist_policy, 2)


if __name__ == "__main__":
    """test the bloom model policy"""
    test_bloom_model_policy()
