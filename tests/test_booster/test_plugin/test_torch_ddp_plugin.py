import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import TorchDDPPlugin
from colossalai.interface import OptimizerWrapper
from colossalai.testing import rerun_if_address_is_in_use, spawn
from tests.kit.model_zoo import model_zoo


def run_fn(model_fn, data_gen_fn, output_transform_fn):
    plugin = TorchDDPPlugin()
    booster = Booster(plugin=plugin)
    model = model_fn()
    optimizer = SGD(model.parameters(), lr=1e-3)
    criterion = lambda x: x.mean()
    data = data_gen_fn()

    data = {k: v.to('cuda') if torch.is_tensor(v) or 'Tensor' in v.__class__.__name__ else v for k, v in data.items()}

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
    for name, (model_fn, data_gen_fn, output_transform_fn, _) in model_zoo.items():
        if name == 'dlrm_interactionarch':
            continue
        run_fn(model_fn, data_gen_fn, output_transform_fn)
        torch.cuda.empty_cache()


def run_dist(rank, world_size, port):
    # init dist env
    colossalai.launch(config=dict(), rank=rank, world_size=world_size, port=port, host='localhost')
    check_torch_ddp_plugin()


@rerun_if_address_is_in_use()
def test_torch_ddp_plugin():
    spawn(run_dist, 2)
