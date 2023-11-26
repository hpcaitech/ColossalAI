import pytest
import torch

import colossalai
from colossalai.legacy.amp import AMP_TYPE
from colossalai.legacy.core import global_context as gpc
from colossalai.testing import DummyDataloader, parameterize, rerun_if_address_is_in_use, spawn
from tests.kit.model_zoo import model_zoo

CONFIG = dict(
    parallel=dict(pipeline=dict(size=1), tensor=dict(size=1, mode=None)), fp16=dict(mode=None), clip_grad_norm=1.0
)


@parameterize("model_name", ["repeated_computed_layers", "resnet18", "repeated_computed_layers"])
@parameterize("amp_mode", [AMP_TYPE.APEX, AMP_TYPE.TORCH, AMP_TYPE.NAIVE, None])
def run_train(model_name, amp_mode):
    # FIXME: test bert
    model_builder, data_gen_fn, *_ = next(iter(model_zoo.get_sub_registry(model_name).values()))
    train_dataloader = DummyDataloader(data_gen_fn)
    criterion = lambda x: x.sum()
    gpc.config.fp16["mode"] = amp_mode

    model = model_builder()
    engine, train_dataloader, *args = colossalai.legacy.initialize(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        criterion=criterion,
        train_dataloader=train_dataloader,
    )

    try:
        engine.train()
        for data in train_dataloader:
            engine.zero_grad()
            data = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
            if criterion:
                output = engine(**data)
                loss = engine.criterion(output)
            else:
                loss = engine(**data)
            engine.backward(loss)
            engine.step()
            break
    except IndexError:
        # if using apex amp, NetWithRepeatedlyComputedLayers will raise an index out of range issue
        # the following check fails in apex
        # if cached_x.grad_fn.next_functions[1][0].variable is not x:
        pass


def run_engine(rank, world_size, port):
    # init dist env
    colossalai.legacy.launch(
        config=CONFIG, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl"
    )
    run_train()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_engine():
    spawn(run_engine, 2)


if __name__ == "__main__":
    test_engine()
