import torch
from torch.fx import symbolic_trace

from colossalai.fx._compatibility import is_compatible_with_meta
from colossalai.fx.passes.meta_info_prop import MetaInfoProp, TensorMetadata
from colossalai.testing import clear_cache_before_run

if is_compatible_with_meta():
    from colossalai.fx.profiler import MetaTensor

BATCH_SIZE = 2
DIM_IN = 4
DIM_OUT = 16


def meta_check(meta_info_spec: TensorMetadata, orig_tensor: torch.Tensor):
    assert meta_info_spec.shape == orig_tensor.shape
    assert meta_info_spec.dtype == orig_tensor.dtype
    assert meta_info_spec.stride == orig_tensor.stride()
    assert meta_info_spec.numel == orig_tensor.numel()


@clear_cache_before_run()
def test_meta_info_prop():
    model = torch.nn.Linear(DIM_IN, DIM_OUT)
    input_sample = torch.rand(BATCH_SIZE, DIM_IN, device="meta")
    if is_compatible_with_meta():
        input_sample = MetaTensor(input_sample, fake_device="cpu")
    orig_output = model(input_sample)
    gm = symbolic_trace(model)
    MetaInfoProp(gm).run(input_sample)
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            meta_check(node.meta["tensor_meta"], input_sample)
        if node.op == "output":
            meta_check(node.meta["tensor_meta"], orig_output)


if __name__ == "__main__":
    test_meta_info_prop()
