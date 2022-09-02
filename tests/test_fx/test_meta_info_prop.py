import torch
import torch.nn as nn
import colossalai
import colossalai.nn as col_nn
from torch.fx import symbolic_trace
from colossalai.fx.passes.meta_info_prop import MetaInfoProp, TensorMetadata

import pytest
try:
    meta_lib = torch.library.Library("aten", "IMPL", "Meta")
    INCOMPATIBLE = False    # version > 1.12.0
except:
    INCOMPATIBLE = True

BATCH_SIZE = 2
DIM_IN = 4
DIM_OUT = 16


def meta_check(meta_info_spec: TensorMetadata, orig_tensor: torch.Tensor):
    assert meta_info_spec.shape == orig_tensor.shape
    assert meta_info_spec.dtype == orig_tensor.dtype
    assert meta_info_spec.stride == orig_tensor.stride()
    assert meta_info_spec.numel == orig_tensor.numel()


@pytest.skip.skipif(INCOMPATIBLE, reason='torch version is lower than 1.12.0')
def test_meta_info_prop():
    model = torch.nn.Linear(DIM_IN, DIM_OUT)
    input_sample = torch.rand(BATCH_SIZE, DIM_IN, device='meta')
    orig_output = model(input_sample)
    gm = symbolic_trace(model)
    MetaInfoProp(gm).run(input_sample)
    for node in gm.graph.nodes:
        if node.op == 'placeholder':
            meta_check(node.meta['tensor_meta'], input_sample)
        if node.op == 'output':
            meta_check(node.meta['tensor_meta'], orig_output)


if __name__ == '__main__':
    test_meta_info_prop()
