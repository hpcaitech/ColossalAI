import os
from functools import partial

import pytest
import torch
import torch.distributed.rpc as rpc
from rpc_test_utils import DAG_MLP, MLP
from torch._C._distributed_rpc import _is_current_rpc_agent_set

from colossalai.fx import ColoTracer
from colossalai.fx.passes.adding_split_node_pass import balanced_split_pass, split_with_split_nodes_pass
from colossalai.legacy import launch
from colossalai.legacy.pipeline.middleware.adaptor import get_fx_topology
from colossalai.legacy.pipeline.pipeline_process_group import ppg
from colossalai.legacy.pipeline.rpc._pipeline_schedule import OneFOneBPipelineEngine
from colossalai.logging import disable_existing_loggers
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn

# global variable for model created
batch_size = 16
dim = 10
rpc_is_initialized = _is_current_rpc_agent_set


def create_partition_module(pp_rank: int, stage_num: int, model, data_kwargs):
    model.eval()
    tracer = ColoTracer()
    meta_args = {k: v.to("meta") for k, v in data_kwargs.items()}
    graph = tracer.trace(root=model, meta_args=meta_args)
    gm = torch.fx.GraphModule(model, graph, model.__class__.__name__)
    annotated_model = balanced_split_pass(gm, stage_num)
    top_module, split_submodules = split_with_split_nodes_pass(annotated_model, merge_output=True)
    topo = get_fx_topology(top_module)
    for submodule in split_submodules:
        if isinstance(submodule, torch.fx.GraphModule):
            setattr(submodule, "_topo", topo)
    return split_submodules[pp_rank + 1]


def partition(model, data_kwargs: dict, pp_rank: int, chunk: int, stage_num: int):
    torch.manual_seed(1024)
    partition = create_partition_module(pp_rank, stage_num, model, data_kwargs)
    return partition


def run_master(model_cls, world_size, forward_only):
    torch.manual_seed(100)

    epoch = 3
    device = "cuda"
    stage_num = world_size
    chunk = 1
    num_microbatches = 8
    use_checkpoint = "store_true"

    if model_cls == MLP:

        def data_gen():
            x = torch.zeros((batch_size, dim))
            kwargs = dict(x=x)
            return kwargs

        model = model_cls(dim, stage_num * 3)
        if forward_only:
            labels = None
        else:
            labels = 1
    elif model_cls == DAG_MLP:

        def data_gen():
            x = torch.zeros((batch_size, dim))
            y = torch.zeros((batch_size, dim))
            kwargs = dict(x=x, y=y)
            return kwargs

        model = model_cls(dim, stage_num * 3)
        if forward_only:
            labels = None
        else:
            labels = 1
    else:
        pass

    data_kwargs = data_gen()

    engine = OneFOneBPipelineEngine(
        partition_fn=partial(partition, model, data_kwargs),
        stage_num=stage_num,
        num_microbatches=num_microbatches,
        device=device,
        chunk=chunk,
        checkpoint=use_checkpoint,
    )
    if not forward_only:
        engine.initialize_optimizer(getattr(torch.optim, "SGD"), lr=1e-3)

    for _ in range(epoch):
        input_x = torch.randn((batch_size, dim), device=device)
        input_y = torch.randn((batch_size, dim), device=device)
        logits = engine.forward_backward({"x": input_x, "y": input_y}, labels=labels, forward_only=forward_only)


def run_worker(rank, world_size, port, model_cls, forward_only, master_func):
    master_addr = "localhost"
    master_port = 29020
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    disable_existing_loggers()

    launch(dict(), rank, world_size, master_addr, master_port, "nccl", verbose=False)
    ppg.set_global_info(
        rank=rank, world_size=world_size, dp_degree=1, tp_degree=1, num_worker_threads=128, device="cuda"
    )

    # in rpc mode, only rank 0 is needed to be coded
    if rank == 0:
        master_func(model_cls, world_size, forward_only)
    # barrier here
    if rpc_is_initialized():
        rpc.shutdown()


@pytest.mark.skip("skip due to CI torch version 1.11")
@parameterize("model_cls", [MLP, DAG_MLP])
@parameterize("forward_only", [True, False])
@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_pp_middleware_fwd(model_cls, forward_only):
    world_size = 4
    master_func = run_master
    spawn(
        run_worker,
        world_size,
        model_cls=model_cls,
        forward_only=forward_only,
        master_func=master_func,
    )


if __name__ == "__main__":
    test_pp_middleware_fwd()
