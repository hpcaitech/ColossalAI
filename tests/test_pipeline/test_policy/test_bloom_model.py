import pytest
import torch
import torch.distributed as dist
from transformers.models.bloom import BloomConfig, BloomModel

import colossalai
from colossalai.cluster import ProcessGroupMesh
from colossalai.pipeline.policy.bloom import BloomModelPolicy, bloom_model_forward
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.testing import rerun_if_address_is_in_use, spawn


def check_bloom_model_forward():
    # create a BloomModel
    configuration = BloomConfig()
    model = BloomModel(configuration)
    DP_DIM, PP_DIM = 0, 1
    DP_SIZE, PP_SIZE = 2, 2
    RANK_TO_COORDINATE = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 0),
        3: (1, 1),
    }
    PP_RANKS_IN_GROUP = {
        0: [0, 1],
        1: [0, 1],
        2: [2, 3],
        3: [2, 3],
    }
    pg_mesh = ProcessGroupMesh(DP_SIZE, PP_SIZE)
    # print(pg_mesh)

    stage_manager = PipelineStageManager(pg_mesh, PP_DIM)
    rank = dist.get_rank()
    # print(rank)

    x = torch.randint(0, 1000, (2, 3))
    hidden_states = torch.randint(0, 1000, (2, 3, 64)).to(torch.float32)
    if stage_manager.is_first_stage():
        attention_mask = torch.ones_like(x)
        output = bloom_model_forward(self=model,
                                     input_ids=x,
                                     attention_mask=attention_mask,
                                     stage_manager=stage_manager)
        print(output[0].shape)
        assert output[0].shape == (2, 3, 64)
        print('start the training')
    else:
        attention_mask = torch.ones((2, 3))
        output = bloom_model_forward(self=model,
                                     hidden_states=hidden_states,
                                     attention_mask=attention_mask,
                                     stage_manager=stage_manager)
        print(output[0].shape)
        assert output[0].shape == (2, 3, 64)
        print('end the training')
        print(output)

    # assert output[1].shape == (2, 768)


def check_bloom_model_policy():
    # create a BloomModel
    configuration = BloomConfig()
    model = BloomModel(configuration)
    DP_DIM, PP_DIM = 0, 1
    DP_SIZE, PP_SIZE = 2, 2
    RANK_TO_COORDINATE = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 0),
        3: (1, 1),
    }
    PP_RANKS_IN_GROUP = {
        0: [0, 1],
        1: [0, 1],
        2: [2, 3],
        3: [2, 3],
    }
    pg_mesh = ProcessGroupMesh(DP_SIZE, PP_SIZE)
    # print(pg_mesh)

    stage_manager = PipelineStageManager(pg_mesh, PP_DIM)
    rank = dist.get_rank()

    model_policy = BloomModelPolicy(stage_manager=stage_manager, num_layers=len(model.h), num_stages=2)
    assert model_policy.layers_per_stage == [1, 1]
    layers = model_policy.get_hold_layers(model)
    for layer in layers:
        print(layer)


def run_dist_model(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, port=port, host='localhost')
    check_bloom_model_forward()


def run_dist_policy(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, port=port, host='localhost')
    check_bloom_model_policy()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_bloom_model_forward():
    spawn(run_dist_model, 4)


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_bloom_model_policy():
    spawn(run_dist_policy, 4)


if __name__ == "__main__":
    """test the bloom model forward and bloom model policy"""
    test_bloom_model_forward()
    test_bloom_model_policy()
