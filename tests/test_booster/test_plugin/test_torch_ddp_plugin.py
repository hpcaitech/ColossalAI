from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import TorchDDPPlugin
from colossalai.interface import OptimizerWrapper
from colossalai.testing import clear_cache_before_run, rerun_if_address_is_in_use, spawn
from tests.kit.model_zoo import COMMON_MODELS, IS_FAST_TEST, model_zoo


@clear_cache_before_run()
def run_fn(model_fn, data_gen_fn, output_transform_fn):
    plugin = TorchDDPPlugin()
    booster = Booster(plugin=plugin)
    model = model_fn()
    optimizer = SGD(model.parameters(), lr=1e-3)
    criterion = lambda x: x.mean()
    data = data_gen_fn()

    data = {k: v.to("cuda") if torch.is_tensor(v) or "Tensor" in v.__class__.__name__ else v for k, v in data.items()}

    model, optimizer, criterion, _, _ = booster.boost(model, optimizer, criterion)

    assert isinstance(model.module, DDP)
    assert isinstance(optimizer, OptimizerWrapper)

    output = model(**data)
    output = output_transform_fn(output)
    output_key = list(output.keys())[0]
    loss = criterion(output[output_key])

    booster.backward(loss, optimizer)
    optimizer.clip_grad_by_norm(1.0)
    optimizer.step()


def check_torch_ddp_plugin():
    if IS_FAST_TEST:
        registry = model_zoo.get_sub_registry(COMMON_MODELS)
    else:
        registry = model_zoo

    for name, (model_fn, data_gen_fn, output_transform_fn, _, _) in registry.items():
        if name in ("dlrm_interactionarch", "transformers_mixtral") or name.startswith("simple_"):
            continue
        run_fn(model_fn, data_gen_fn, output_transform_fn)
        torch.cuda.empty_cache()


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(1))

    def forward(self, x):
        return self.weight * x


def check_torch_ddp_no_sync():
    plugin = TorchDDPPlugin()
    booster = Booster(plugin=plugin)

    model = DummyModel()
    criterion = lambda x: x.mean()
    optimizer = SGD(model.parameters(), lr=1e-3)
    # create a custom dataset with 0 to 10
    dataset = torch.arange(0, 10)
    train_dataloader = plugin.prepare_dataloader(dataset, batch_size=2)
    model, optimizer, criterion, train_dataloader, _ = booster.boost(
        model, optimizer, criterion, dataloader=train_dataloader
    )

    def fwd_bwd():
        output = model(batch.cuda())
        loss = criterion(output)
        booster.backward(loss, optimizer)

    def get_grad_set_over_all_ranks():
        for p in model.parameters():
            # grad shape is (1, )
            assert p.grad.shape == (1,)
            grad_list = [torch.empty_like(p.grad) for _ in range(dist.get_world_size())]
            dist.all_gather(grad_list, p.grad)
            # get grad set of all ranks
            grad_set = set([grad.item() for grad in grad_list])
            # as the model only has one parameter, we can return here
            return grad_set

    for i, batch in enumerate(train_dataloader):
        if i > 1:
            # only check the first two batches
            break
        # no_sync for the first batch, sync for the second batch
        ctx = booster.no_sync(model) if i == 0 else nullcontext()
        with ctx:
            fwd_bwd()
        grad_set = get_grad_set_over_all_ranks()
        # for the first batch, all ranks should have different grads
        # for the second batch, as grad is synchronized,all ranks should have the same grads
        target_num_different_grad = dist.get_world_size() if i == 0 else 1
        assert len(grad_set) == target_num_different_grad


def run_dist(rank, world_size, port):
    # init dist env
    colossalai.launch(rank=rank, world_size=world_size, port=port, host="localhost")
    check_torch_ddp_plugin()
    check_torch_ddp_no_sync()


@rerun_if_address_is_in_use()
def test_torch_ddp_plugin():
    spawn(run_dist, 2)
