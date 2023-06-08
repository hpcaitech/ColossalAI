import pytest
import torch
import torch.nn.functional as F

import colossalai
from colossalai.logging import disable_existing_loggers
from colossalai.shardformer.policies.basepolicy import Col_Layer, Layer, Row_Layer
from colossalai.shardformer.shard.shard_config import ShardConfig
from colossalai.shardformer.shard.slicer import Slicer
from colossalai.testing import rerun_if_address_is_in_use, spawn

CONFIG = dict(parallel=dict(data=1, pipeline=1, tensor=dict(size=2, mode='1d')),)


def check_slicer(rank, world_size, port, in_feature, out_feature):
    disable_existing_loggers()
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, port=port, host='localhost', backend='nccl')
    # initialize slicer
    shardconfig = ShardConfig(rank=rank, world_size=world_size)
    slicer = Slicer(shardconfig)
    # initialize test data
    weight = torch.randn(in_feature, out_feature)
    bias = torch.randn(out_feature)
    policy_layer_cls_list = [Layer, Col_Layer, Row_Layer]
    n_cast_list = [None, 2, 3, 4]
    # weight and bias
    for n_cast in n_cast_list:
        sliced_weight, sliced_bias = slicer.slice_weight_bias(weight, bias, policy_layer_cls=Layer, n_cast=n_cast)
        expected_sliced_weight = weight
        expected_sliced_bias = bias
        assert torch.equal(
            sliced_weight, expected_sliced_weight
        ), f"In Layer case, weight: sliced_weight is not equal to expected_sliced_weight\norg:{weight}\nsliced:{sliced_weight}\nexpected:{expected_sliced_weight}"
        assert torch.equal(
            sliced_bias, expected_sliced_bias
        ), f"In Layer case, bias: sliced_bias is not equal to expected_sliced_bias\norg:{bias}\nsliced:{sliced_weight}\nexpected:{expected_sliced_weight}"

        sliced_weight, sliced_bias = slicer.slice_weight_bias(weight, bias, policy_layer_cls=Col_Layer, n_cast=n_cast)
        if (n_cast is None):
            expected_sliced_weight = weight.chunk(world_size, dim=0)[rank]
            expected_sliced_bias = bias.chunk(world_size)[rank]
        else:
            chunks = weight.chunk(world_size * n_cast, dim=0)
            expected_sliced_weight = torch.cat([chunks[i] for i in range(rank, n_cast * world_size, world_size)], dim=0)
            chunks = bias.chunk(world_size * n_cast, dim=0)
            expected_sliced_bias = torch.cat([chunks[i] for i in range(rank, n_cast * world_size, world_size)])
        assert torch.equal(
            sliced_weight, expected_sliced_weight
        ), f"In Col_Layer {n_cast} cast case, weight: sliced_weight is not equal to expected_sliced_weight\norg:{weight}\nsliced:{sliced_weight}\nexpected:{expected_sliced_weight}"
        assert torch.equal(
            sliced_bias, expected_sliced_bias
        ), f"In Col_Layer {n_cast} cast case, bias: sliced_bias is not equal to expected_sliced_bias\norg:{bias}\nsliced:{sliced_bias}\nexpected:{expected_sliced_bias}"

        sliced_weight, sliced_bias = slicer.slice_weight_bias(weight, bias, policy_layer_cls=Row_Layer, n_cast=n_cast)
        if (n_cast is None):
            expected_sliced_weight = weight.chunk(world_size, dim=1)[rank]
            expected_sliced_bias = bias
        else:
            chunks = weight.chunk(world_size * n_cast, dim=1)
            expected_sliced_weight = torch.cat([chunks[i] for i in range(rank, n_cast * world_size, world_size)], dim=1)
            expected_sliced_bias = bias
        assert torch.equal(
            sliced_weight, expected_sliced_weight
        ), f"In Row_Layer {n_cast} cast case, weight: sliced_weight is not equal to expected_sliced_weight\norg:{weight}\nsliced:{sliced_weight}\nexpected:{expected_sliced_weight}"
        assert torch.equal(
            sliced_bias, expected_sliced_bias
        ), f"In Row_Layer {n_cast} cast case, bias: sliced_bias is not equal to expected_sliced_bias\norg:{bias}\nsliced:{sliced_weight}\nexpected:{expected_sliced_weight}"


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_slicer():
    args = dict(in_feature=24, out_feature=48)
    spawn(check_slicer, nprocs=2, in_feature=args['in_feature'], out_feature=args['out_feature'])


if __name__ == '__main__':
    test_slicer()
