import pytest
import torch

import colossalai
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.zero import GeminiDDP
from colossalai.zero.gemini.chunk import search_chunk_configuration
from tests.components_to_test.registry import non_distributed_component_funcs


@parameterize('placement_policy', ['cuda', 'cpu'])
@parameterize('model_name', ['gpt2', 'bert'])
def exam_state_dict(placement_policy, model_name: str):
    get_components_func = non_distributed_component_funcs.get_callable(model_name)
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()

    model = model_builder()

    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2

    config_dict, *_ = search_chunk_configuration(model, search_range_m=1, search_interval=100)
    model = GeminiDDP(model, config_dict, placement_policy=placement_policy)
    model.train()

    zero_dict = model.state_dict(only_rank_0=False)
    accumulated_keys = set()
    # ensure number of shards > 1
    for shard, _ in model.state_dict_shard(max_shard_size=(model_size / 3), only_rank_0=False):
        for key, value in shard.items():
            assert key not in accumulated_keys, f"key `{key}` is duplicated."
            accumulated_keys.add(key)
            assert key in zero_dict, f"{key} not in ZeRO dictionary."
            assert torch.equal(value, zero_dict[key]), f"{key} not equal."


def run_dist(rank, world_size, port):
    config = {}
    colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    exam_state_dict()


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 4])
@rerun_if_address_is_in_use()
def test_zero_ddp_state_dict_shard(world_size):
    spawn(run_dist, world_size)


if __name__ == '__main__':
    test_zero_ddp_state_dict_shard(1)
