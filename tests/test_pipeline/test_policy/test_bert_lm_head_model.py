import pytest
import torch
import torch.distributed as dist
from transformers.models.bert import BertConfig
from transformers.models.bert.modeling_bert import BertLMHeadModel

import colossalai
from colossalai.cluster import ProcessGroupMesh
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer.policies.base_policy import Policy
from colossalai.shardformer.policies.bert import BertLMHeadModelPolicy, bert_lm_head_model_forward
from colossalai.shardformer.shard import ShardConfig
from colossalai.testing import rerun_if_address_is_in_use, spawn


def check_bert_lm_head_model_forward():
    configuration = BertConfig()
    model = BertLMHeadModel(configuration)
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
    layers_per_stage = Policy.distribute_layers(len(model.bert.encoder.layer), 2)
    stage_index = Policy.get_stage_index(layers_per_stage, stage_manager.stage)
    x = torch.randint(0, 1000, (2, 3))
    hidden_states = torch.randint(0, 1000, (2, 3, 768)).to(torch.float32)
    if stage_manager.stage == 0:
        attention_mask = torch.ones_like(x)

        output = bert_lm_head_model_forward(self=model,
                                            input_ids=x,
                                            attention_mask=attention_mask,
                                            stage_manager=stage_manager,
                                            stage_index=stage_index)
        print(output['hidden_states'].shape)
        assert output['hidden_states'].shape == (2, 3, 768)

    else:
        attention_mask = torch.ones((2, 3))
        output = bert_lm_head_model_forward(self=model,
                                            hidden_states=hidden_states,
                                            attention_mask=attention_mask,
                                            stage_manager=stage_manager,
                                            stage_index=stage_index)
        print(output[0].shape)
        assert output[0].shape == (2, 3, 30522)

    # assert output[1].shape == (2, 768)


def check_bert_lmhead_policy():
    configuration = BertConfig()
    model = BertLMHeadModel(configuration)
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

    model_policy = BertLMHeadModelPolicy()
    model_policy.set_model(model)
    model_config = ShardConfig(pipeline_stage_manager=stage_manager, enable_tensor_parallelism=False)
    model_policy.set_shard_config(model_config)
    layers = model_policy.get_held_layers()

    assert layers is not None


def run_dist_model(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, port=port, host='localhost')
    check_bert_lm_head_model_forward()


def run_dist_policy(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, port=port, host='localhost')
    check_bert_lmhead_policy()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_bert_lm_head_model_forward():
    spawn(run_dist_model, 4)


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_bert_lmhead_policy():
    spawn(run_dist_policy, 4)


if __name__ == "__main__":
    """test the bert for pretraining model forward and bert for pretraining model policy"""
    test_bert_lm_head_model_forward()
    test_bert_lmhead_policy()
