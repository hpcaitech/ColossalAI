import pytest
import torch

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.testing import rerun_if_address_is_in_use, spawn
from colossalai.zero.gemini.chunk import init_chunk_manager, search_chunk_configuration
from tests.kit.model_zoo import model_zoo


def exam_search_chunk_size():
    model_builder, data_gen_fn, output_transform_fn, *_ = next(
        iter(model_zoo.get_sub_registry("transformers_gpt_lm").values())
    )

    # make sure torch_model and model has the same parameter values
    model = model_builder()
    config_dict, *_ = search_chunk_configuration(
        model, search_range_m=1, search_interval=128, min_chunk_size_m=0, filter_exlarge_params=True
    )

    for key in config_dict:
        chunk_size = config_dict[key]["chunk_size"]
        assert chunk_size == 527872


def exam_chunk_manager():
    world_size = torch.distributed.get_world_size()

    model_builder, data_gen_fn, output_transform_fn, *_ = next(
        iter(model_zoo.get_sub_registry("transformers_gpt_lm").values())
    )

    sharded_ddp_model = model_builder()
    chunk_manager = init_chunk_manager(
        sharded_ddp_model,
        get_accelerator().get_current_device(),
        hidden_dim=128,
        search_range_m=1,
        min_chunk_size_m=0,
        filter_exlarge_params=True,
        strict_ddp_flag=True,
    )
    config_dict = chunk_manager.dp_degree_chunk_size_dict
    assert len(config_dict) == 1
    assert config_dict[world_size] == 527872


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    exam_search_chunk_size()
    exam_chunk_manager()


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [1, 4])
@rerun_if_address_is_in_use()
def test_search(world_size):
    spawn(run_dist, world_size)


if __name__ == "__main__":
    test_search(4)
