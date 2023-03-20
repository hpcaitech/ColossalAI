import torch
from torch.optim import Adam

from colossalai.booster.mixed_precision import FP16TorchMixedPrecision
from tests.kit.model_zoo import model_zoo


def test_torch_amp():
    for name, (model_fn, data_gen_fn, output_transform_fn, _) in model_zoo.items():
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
