from copy import deepcopy

import torch
import torch.nn as nn

from colossalai.shardformer.layer import Linear1D_Col, Linear1D_Row

from ..registry import model_zoo

_BS = 16
_IN_DIM = 32
_HID_DIM = 128


class Net(nn.Module):
    def __init__(self, in_dim=_IN_DIM, hid_dim=_HID_DIM, identity=False, dtype=torch.float32):
        super().__init__()
        if identity:
            self.fc0 = nn.Identity()
        else:
            self.fc0 = nn.Linear(in_dim, in_dim).to(dtype=dtype)

        self.fc1 = nn.Linear(in_dim, hid_dim).to(dtype=dtype)
        self.fc2 = nn.Linear(hid_dim, in_dim).to(dtype=dtype)

    def forward(self, x):
        return self.fc2(self.fc1(self.fc0(x)))


class TPNet(nn.Module):
    def __init__(
        self,
        fc0=nn.Linear(_IN_DIM, _IN_DIM),
        fc1=nn.Linear(_IN_DIM, _HID_DIM),
        fc2=nn.Linear(_HID_DIM, _IN_DIM),
        tp_group=None,
        dtype=torch.float32,
    ):
        super().__init__()
        self.fc0 = deepcopy(fc0)
        self.fc1 = Linear1D_Col.from_native_module(
            deepcopy(fc1), process_group=tp_group, gather_output=False, overlap=True, dtype=dtype
        )
        self.fc2 = Linear1D_Row.from_native_module(
            deepcopy(fc2), process_group=tp_group, parallel_input=True, dtype=dtype
        )

    def forward(self, x):
        return self.fc2(self.fc1(self.fc0(x)))


def data_gen():
    return torch.randn(_BS, _IN_DIM)


def output_transform(x: torch.Tensor):
    return x


model_zoo.register(name="simple_mlp", model_fn=Net, data_gen_fn=data_gen, output_transform_fn=output_transform)
model_zoo.register(name="simple_tp_mlp", model_fn=TPNet, data_gen_fn=data_gen, output_transform_fn=output_transform)
