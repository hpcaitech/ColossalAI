'''
In the test policy we only test policy: held layers and others, as the tests for forward logic are done in test_shardformer/test_model
'''

import pytest
import torch.distributed as dist
from transformers.models.bert.modeling_bert import BertModel

import colossalai
from colossalai.cluster import ProcessGroupMesh
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer.policies.auto_policy import get_autopolicy
from colossalai.shardformer.policies.base_policy import Policy
from colossalai.shardformer.policies.bert import BertModelPolicy
from colossalai.shardformer.shard import ShardConfig
from colossalai.testing import rerun_if_address_is_in_use, spawn
from tests.kit.model_zoo import model_zoo
from tests.test_shardformer.test_model._utils import build_pipeline_model


def check_bert_model_policy():
    PP_DIM = 0
    PP_SIZE = 2
    pg_mesh = ProcessGroupMesh(PP_SIZE)
    # print(pg_mesh)
    stage_manager = PipelineStageManager(pg_mesh, PP_DIM)
    sub_model_zoo = model_zoo.get_sub_registry('transformers_bert')
    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        org_model = model_fn()
        policy = get_autopolicy(org_model)
        policy.set_model(org_model)
        model_config = ShardConfig(pipeline_stage_manager=stage_manager, enable_tensor_parallelism=False)
        policy.set_shard_config(model_config)
        layers = policy.get_held_layers()
        if stage_manager.is_first_stage():
            assert len(layers) == 1 + 1
        else:
            if name == "transformers_bert":
                assert len(layers) == 1 + 1
                continue
            if name in [
                    "transformers_bert_for_sequence_classification", "transformers_bert_for_token_classification",
                    "transformers_bert_for_mcq"
            ]:
                assert len(layers) == 1 + 3
                continue
            else:
                assert len(layers) == 1 + 2


def run_dist_policy(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, port=port, host='localhost')
    check_bert_model_policy()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_bert_model_policy():
    spawn(run_dist_policy, 2)


if __name__ == "__main__":
    """test the bert model policy"""
    test_bert_model_policy()
