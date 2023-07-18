import torch
from torch.optim import Adam

import colossalai
from colossalai.booster.mixed_precision import FP16TorchMixedPrecision
from colossalai.testing import rerun_if_address_is_in_use, spawn
from tests.kit.model_zoo import model_zoo


def run_torch_amp(rank, world_size, port):
    # init dist env
    colossalai.launch(config=dict(), rank=rank, world_size=world_size, port=port, host='localhost')
    sub_model_zoo = model_zoo.get_sub_registry('timm')
    for name, (model_fn, data_gen_fn, output_transform_fn, _, _) in sub_model_zoo.items():
        # dlrm_interactionarch has not parameters, so skip
        if name == 'dlrm_interactionarch':
            continue

        model = model_fn().cuda()
        optimizer = Adam(model.parameters(), lr=1e-3)
        criterion = lambda x: x.mean()
        data = data_gen_fn()
        data = {
            k: v.to('cuda') if torch.is_tensor(v) or 'Tensor' in v.__class__.__name__ else v for k, v in data.items()
        }
        mixed_precision = FP16TorchMixedPrecision()
        model, optimizer, criterion = mixed_precision.configure(model, optimizer, criterion)
        output = model(**data)
        output = output_transform_fn(output)
        output_key = list(output.keys())[0]
        loss = criterion(output[output_key])
        optimizer.backward(loss)
        optimizer.clip_grad_by_norm(1.0)
        optimizer.step()
        del model, optimizer, criterion, data, output, mixed_precision


@rerun_if_address_is_in_use()
def test_torch_ddp_plugin():
    spawn(run_torch_amp, 1)
