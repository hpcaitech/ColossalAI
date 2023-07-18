import pytest
import torch
from packaging import version
from torch.optim import SGD

import colossalai
from colossalai.booster import Booster

if version.parse(torch.__version__) >= version.parse('1.12.0'):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from colossalai.booster.plugin import TorchFSDPPlugin

from colossalai.interface import OptimizerWrapper
from colossalai.testing import rerun_if_address_is_in_use, spawn
from tests.kit.model_zoo import model_zoo


# test baisc fsdp function
def run_fn(model_fn, data_gen_fn, output_transform_fn):
    plugin = TorchFSDPPlugin()
    booster = Booster(plugin=plugin)
    model = model_fn()
    optimizer = SGD(model.parameters(), lr=1e-3)
    criterion = lambda x: x.mean()
    data = data_gen_fn()

    data = {k: v.to('cuda') if torch.is_tensor(v) or 'Tensor' in v.__class__.__name__ else v for k, v in data.items()}

    model, optimizer, criterion, _, _ = booster.boost(model, optimizer, criterion)

    assert isinstance(model.module, FSDP)
    assert isinstance(optimizer, OptimizerWrapper)

    output = model(**data)
    output = output_transform_fn(output)
    output_key = list(output.keys())[0]
    loss = criterion(output[output_key])

    booster.backward(loss, optimizer)
    optimizer.clip_grad_by_norm(1.0)
    optimizer.step()


def check_torch_fsdp_plugin():
    for name, (model_fn, data_gen_fn, output_transform_fn, _, _) in model_zoo.items():
        if any(element in name for element in [
                'diffusers', 'deepfm_sparsearch', 'dlrm_interactionarch', 'torchvision_googlenet',
                'torchvision_inception_v3'
        ]):
            continue
        run_fn(model_fn, data_gen_fn, output_transform_fn)
        torch.cuda.empty_cache()


def run_dist(rank, world_size, port):
    # init dist env
    colossalai.launch(config=dict(), rank=rank, world_size=world_size, port=port, host='localhost')
    check_torch_fsdp_plugin()


@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.12.0'), reason="requires torch1.12 or higher")
@rerun_if_address_is_in_use()
def test_torch_fsdp_plugin():
    spawn(run_dist, 2)
