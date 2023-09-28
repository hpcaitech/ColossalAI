import torch
import torch.nn as nn

from colossalai.shardformer.shard.utils import set_tensors_to_none


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(1, 2), nn.Linear(2, 3))
        self.out = nn.Linear(3, 1)


def test_release_layer():
    orig_cuda_allocated = torch.cuda.memory_allocated()
    model = Net().cuda()
    set_tensors_to_none(model, exclude={model.layers[0]})
    assert model.layers[1].weight is None
    assert model.layers[1].bias is None
    assert model.out.weight is None
    assert model.out.bias is None
    set_tensors_to_none(model)
    assert model.layers[0].weight is None
    assert model.layers[0].bias is None
    assert len(list(model.parameters())) == 0
    assert torch.cuda.memory_allocated() == orig_cuda_allocated
