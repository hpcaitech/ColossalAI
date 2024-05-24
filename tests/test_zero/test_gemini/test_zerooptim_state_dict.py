import pytest
import torch
import torch.distributed as dist

import colossalai
from colossalai.nn.optimizer import HybridAdam
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.utils import set_seed
from colossalai.zero import GeminiDDP, GeminiOptimizer
from colossalai.zero.gemini.chunk import search_chunk_configuration
from tests.kit.model_zoo import model_zoo

PLACEMENT_CONFIGS = [
    {"placement_policy": "static", "shard_param_frac": 0.0, "offload_optim_frac": 0.0},  # zero2
    {"placement_policy": "static", "shard_param_frac": 0.0, "offload_optim_frac": 1.0},  # zero2-offload
    {"placement_policy": "static", "shard_param_frac": 0.0, "offload_optim_frac": 0.5},  # zero2-offload-half
    {"placement_policy": "auto"},
]


@parameterize("placement_config", PLACEMENT_CONFIGS)
@parameterize("keep_gathered", [True, False])
def exam_zero_optim_state_dict(placement_config, keep_gathered):
    set_seed(431)
    model_builder, data_gen_fn, output_transform_fn, *_ = next(
        iter(model_zoo.get_sub_registry("transformers_gpt_lm").values())
    )

    model = model_builder()

    set_seed(451)

    world_size = torch.distributed.get_world_size()
    config_dict, *_ = search_chunk_configuration(model, search_range_m=1, search_interval=100)
    config_dict[world_size]["chunk_size"] = 5000
    config_dict[world_size]["keep_gathered"] = keep_gathered

    model = GeminiDDP(model, config_dict, **placement_config, pin_memory=True)

    optimizer = HybridAdam(model.parameters())
    optim = GeminiOptimizer(optimizer, model, initial_scale=32)  # initialize the link between chunk16 and chunk32

    set_seed(dist.get_rank() * 3 + 128)
    model.train()
    data = data_gen_fn()
    data = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()}

    optim.zero_grad()
    outputs = model(**data)
    outputs = output_transform_fn(outputs)
    loss = next(iter(outputs.values())).sum()
    optim.backward(loss)
    optim.step()

    optim_state_dict = optim.state_dict()
    optim.load_state_dict(optim_state_dict)
    new_state = optim.state_dict()["state"]
    org_state = optim_state_dict["state"]

    for k, v in org_state.items():
        w = new_state[k]
        for n, m in v.items():
            if isinstance(m, torch.Tensor):
                o = w[n]
                assert torch.equal(m, o)
            else:
                assert m == w[n]


def run_dist(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    exam_zero_optim_state_dict()


@pytest.mark.skip
@pytest.mark.dist
@pytest.mark.parametrize("world_size", [1, 4])
@rerun_if_address_is_in_use()
def test_zero_optim(world_size):
    spawn(run_dist, world_size)


if __name__ == "__main__":
    test_zero_optim(1)
