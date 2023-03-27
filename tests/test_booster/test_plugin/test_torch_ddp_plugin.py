from functools import partial

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import TorchDDPPlugin
from colossalai.interface import OptimizerWrapper
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils import free_port
from tests.kit.model_zoo import model_zoo


def check_torch_ddp_plugin():
    plugin = TorchDDPPlugin()
    booster = Booster(plugin=plugin)

    for name, (model_fn, data_gen_fn, output_transform_fn, _) in model_zoo.items():
        if name == 'dlrm_interactionarch':
            continue

        model = model_fn()
        optimizer = SGD(model.parameters(), lr=1e-3)
        criterion = lambda x: x.mean()
        data = data_gen_fn()

        data = {
            k: v.to('cuda') if torch.is_tensor(v) or 'Tensor' in v.__class__.__name__ else v for k, v in data.items()
        }

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


def check_dataloader_sharding():
    plugin = TorchDDPPlugin()

    # create a custom dasetset with 0 to 10
    dataset = torch.utils.data.TensorDataset(torch.arange(0, 10))
    train_dataloader = plugin.prepare_train_dataloader(dataset, batch_size=2)

    # get the first batch of data
    batch = next(iter(train_dataloader))[0].cuda()
    is_rank_0 = dist.get_rank() == 0

    if is_rank_0:
        batch_to_compare = batch.clone()
    else:
        batch_to_compare = batch
    # pass to the rank 1 value to rank 0
    dist.broadcast(batch_to_compare, src=1)

    # compare on rank 0
    if is_rank_0:
        assert not torch.equal(batch,
                               batch_to_compare), 'Same number was found across ranks but expected it to be different'


def run_dist(rank, world_size, port):
    # init dist env
    colossalai.launch(config=dict(), rank=rank, world_size=world_size, port=port, host='localhost')
    check_dataloader_sharding()
    check_torch_ddp_plugin()


@rerun_if_address_is_in_use()
def test_torch_ddp_plugin():
    world_size = 2
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)
