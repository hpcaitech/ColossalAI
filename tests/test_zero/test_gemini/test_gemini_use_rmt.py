import pytest
import torch
import torch.distributed as dist

import colossalai
from colossalai.testing import DummyDataloader, parameterize, rerun_if_address_is_in_use, spawn
from colossalai.utils import set_seed
from colossalai.zero import GeminiDDP
from colossalai.zero.gemini.chunk import search_chunk_configuration
from colossalai.zero.gemini.memory_tracer.runtime_mem_tracer import RuntimeMemTracer
from tests.kit.model_zoo import model_zoo, run_fwd_bwd

# run gemini use the runtime memory tracer


@parameterize("placement_policy", ["auto"])
@parameterize("keep_gather", [False])
@parameterize("model_name", ["transformers_bert_for_sequence_classification"])
@parameterize("use_grad_checkpoint", [False, True])
def run_gemini_use_rmt(placement_policy, keep_gather, model_name: str, use_grad_checkpoint: bool = False):
    set_seed(42)
    model_builder, data_gen_fn, output_transform_fn, *_ = next(iter(model_zoo.get_sub_registry(model_name).values()))

    model = model_builder().cuda()
    if use_grad_checkpoint:
        model.gradient_checkpointing_enable()

    print(f"model_name {model_name}")

    runtime_mem_tracer = RuntimeMemTracer(model)
    data = data_gen_fn()
    data = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    run_fwd_bwd(runtime_mem_tracer, data, output_transform_fn, optimizer=runtime_mem_tracer)
    memstats = runtime_mem_tracer.memstats()
    runtime_tracer_non_model_data = runtime_mem_tracer._memstats._non_model_data_cuda_list
    print("runtime tracer non model data points: ", len(runtime_tracer_non_model_data))
    print("runtime tracer: ", runtime_tracer_non_model_data)
    print([memstats.param_used_step(p) for p in model.parameters()])

    if model_name == "repeated_computed_layers":
        for idx, p in enumerate(model.parameters()):
            step_list = memstats.param_used_step(p)
            if idx < 4:
                assert len(step_list) == 4

    if model_name == "repeated_computed_layers":
        for idx, p in enumerate(model.parameters()):
            step_list = memstats.param_used_step(p)
            if idx < 4:
                assert len(step_list) == 4

    world_size = torch.distributed.get_world_size()
    config_dict, *_ = search_chunk_configuration(model, search_range_m=1, search_interval=100)
    config_dict[world_size]["chunk_size"] = 5000
    config_dict[world_size]["keep_gathered"] = keep_gather
    model = GeminiDDP(
        model, chunk_config_dict=config_dict, placement_policy=placement_policy, pin_memory=True, memstats=memstats
    )

    set_seed(dist.get_rank())
    train_dataloader = DummyDataloader(data_gen_fn)
    for i, data in enumerate(train_dataloader):
        # you can only test a single fwd + bwd.
        # after bwd param is grad for Gemini, due to the chunk reuse optimization.
        # print(f'iteration {i}')
        if i > 4:
            break
        data = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()}

        set_seed(42)
        run_fwd_bwd(model, data, output_transform_fn, optimizer=model)

    gemini_non_model_data = model.gemini_manager._mem_stats_collector._memstats.non_model_data_list("cuda")

    # print('gemini non model data:', gemini_non_model_data)

    assert len(gemini_non_model_data) == len(
        runtime_tracer_non_model_data
    ), f"model_name {model_name} {len(gemini_non_model_data)} vs {len(runtime_tracer_non_model_data)}"


def run_dist(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_gemini_use_rmt()


@pytest.mark.skip("this is not used")
@pytest.mark.dist
@pytest.mark.parametrize("world_size", [1, 4])
@rerun_if_address_is_in_use()
def test_gemini_use_rmt(world_size):
    spawn(run_dist, world_size)


if __name__ == "__main__":
    test_gemini_use_rmt(1)
