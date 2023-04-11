from functools import partial

import pytest
import torch
import torch.multiprocessing as mp
from rpc_test_utils import DAG_MLP, MLP

from colossalai import launch
from colossalai.fx import ColoTracer
from colossalai.fx.passes.adding_split_node_pass import balanced_split_pass, split_with_split_nodes_pass
from colossalai.logging import disable_existing_loggers
from colossalai.pipeline.middleware.adaptor import get_fx_topology
from colossalai.pipeline.scheduler import GpipeWorker, PipelineScheduler
from colossalai.testing import parameterize, rerun_if_address_is_in_use


def create_partition_module(pp_rank: int, num_stages: int, model, data_kwargs):
    model.eval()
    tracer = ColoTracer()
    meta_args = {k: v.to('meta') for k, v in data_kwargs.items()}
    graph = tracer.trace(root=model, meta_args=meta_args)
    gm = torch.fx.GraphModule(model, graph, model.__class__.__name__)
    annotated_model = balanced_split_pass(gm, num_stages)
    top_module, split_submodules = split_with_split_nodes_pass(annotated_model, merge_output=True)
    topo = get_fx_topology(top_module)
    for submodule in split_submodules:
        if isinstance(submodule, torch.fx.GraphModule):
            setattr(submodule, '_topo', topo)
    return split_submodules[pp_rank + 1]


def partition(model, data_kwargs: dict, pp_rank: int, num_stages: int):
    torch.manual_seed(1024)
    partition = create_partition_module(pp_rank, num_stages, model, data_kwargs)
    return partition


def get_data_gen(model_cls, forward_only, batch_size, dim, num_stages):
    if model_cls == MLP:

        def data_gen():
            x = torch.zeros((batch_size, dim))
            kwargs = dict(x=x)
            return kwargs

        model = model_cls(dim, num_stages * 3)
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

        model = model_cls(dim, num_stages * 3)
        if forward_only:
            labels = None
        else:
            labels = 1
    else:
        data_gen = None
        model = None
        labels = None
    return model, data_gen, labels


def run_scheduler(rank, model_cls, world_size, forward_only, batch_size, dim):
    torch.manual_seed(100)

    epoch = 1
    device = 'cuda'
    num_stages = world_size
    num_minibatches = 4
    use_checkpoint = False

    model, data_gen, labels = get_data_gen(model_cls, forward_only, batch_size, dim, num_stages)
    data_kwargs = data_gen()

    pp_scheduler = PipelineScheduler(rank=rank,
                                     worker_type=GpipeWorker,
                                     num_stages=num_stages,
                                     num_minibatches=num_minibatches,
                                     partition_fn=partial(partition, model, data_kwargs),
                                     device=device,
                                     checkpoint=use_checkpoint)

    if not forward_only:
        pp_scheduler.initialize_optimizer(getattr(torch.optim, 'SGD'), lr=1e-3)

    for _ in range(epoch):
        input_x = torch.randn((batch_size, dim), device=device)
        input_y = torch.randn((batch_size, dim), device=device)
        pp_scheduler.set_batch({'x': input_x, 'y': input_y})
        if pp_scheduler.is_output_rank():
            pp_scheduler.set_labels(labels)

        logits = pp_scheduler.forward_backward(forward_only=forward_only)


def run_dist(rank, model_cls, forward_only, world_size, batch_size, dim):
    master_addr = 'localhost'
    master_port = 29020
    disable_existing_loggers()
    launch(dict(), rank, world_size, master_addr, master_port, 'nccl', verbose=False)

    run_scheduler(rank, model_cls, world_size, forward_only, batch_size, dim)


def test_scheduler_gpipe(model_cls, forward_only, world_size, batch_size, dim):
    mp.spawn(run_dist, args=(model_cls, forward_only, world_size, batch_size, dim), nprocs=world_size)


@pytest.mark.skip("skip due to CI torch version 1.11")
# @parameterize('model_cls', [MLP, DAG_MLP])
# @parameterize('forward_only', [True, False])
@parameterize('model_cls', [MLP])
@parameterize('forward_only', [True])
@parameterize('world_size', [2])
@parameterize('batch_size', [16])
@parameterize('dim', [10])
@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_scheduler(model_cls, forward_only, world_size, batch_size, dim):
    test_scheduler_gpipe(model_cls, forward_only, world_size, batch_size, dim)


if __name__ == "__main__":
    test_scheduler()
