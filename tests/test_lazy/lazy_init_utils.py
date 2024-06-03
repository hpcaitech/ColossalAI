import random
from copy import deepcopy
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
from packaging import version

from colossalai.device.device_mesh import DeviceMesh
from colossalai.lazy.lazy_init import LazyInitContext, LazyTensor, _MyTensor
from colossalai.tensor.d_tensor import to_global
from colossalai.tensor.d_tensor.layout import Layout
from tests.kit.model_zoo.registry import ModelAttribute

SUPPORT_LAZY = version.parse(torch.__version__) >= version.parse("1.12.0")

# model_fn, data_gen_fn, output_transform_fn, model_attr
TestingEntry = Tuple[Callable[[], torch.nn.Module], Callable[[], dict], Callable[[], dict], Optional[ModelAttribute]]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def assert_model_equal(m1: torch.nn.Module, m2: torch.nn.Module) -> None:
    s1 = m1.state_dict()
    s2 = m2.state_dict()

    assert len(s1) == len(s2), f"len {len(s1)} vs {len(s2)}"

    for (n1, t1), (n2, t2) in zip(s1.items(), s2.items()):
        assert n1 == n2
        assert torch.equal(t1, t2), f"{n1} {t1} vs {t2}"

    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert p1.requires_grad == p2.requires_grad


def assert_forward_equal(
    m1: torch.nn.Module,
    m2: torch.nn.Module,
    data_gen_fn: Callable[[], dict],
    output_transform_fn: Callable[[Any], dict],
) -> None:
    data = data_gen_fn()

    m1.eval()
    m2.eval()
    # run forward
    with torch.no_grad():
        outputs1 = m1(**data)
        outputs2 = m2(**data)

    # compare output
    transformed_out1 = output_transform_fn(outputs1)
    transformed_out2 = output_transform_fn(outputs2)

    assert len(transformed_out1) == len(transformed_out2)

    for key, out1 in transformed_out1.items():
        out2 = transformed_out2[key]
        assert torch.allclose(
            out1, out2, atol=1e-5
        ), f"{m1.__class__.__name__} has inconsistent outputs, {out1} vs {out2}"


def check_lazy_init(
    entry: TestingEntry, seed: int = 42, verbose: bool = False, check_forward: bool = False, default_device: str = "cpu"
) -> None:
    model_fn, data_gen_fn, output_transform_fn, _, model_attr = entry
    _MyTensor._pre_op_fn = lambda *args: set_seed(seed)
    LazyTensor._pre_op_fn = lambda *args: set_seed(seed)
    ctx = LazyInitContext(tensor_cls=_MyTensor, default_device=default_device)
    with ctx:
        model = model_fn()
    ctx = LazyInitContext(default_device=default_device)
    with ctx:
        deferred_model = model_fn()
        copied_deferred_model = deepcopy(deferred_model)
    deferred_model = ctx.materialize(deferred_model, verbose=verbose)
    copied_deferred_model = ctx.materialize(copied_deferred_model, verbose=verbose)
    assert_model_equal(model, deferred_model)
    assert_model_equal(deferred_model, copied_deferred_model)
    if check_forward:
        assert_forward_equal(model, deferred_model, data_gen_fn, output_transform_fn)
        assert_forward_equal(deferred_model, copied_deferred_model, data_gen_fn, output_transform_fn)
    if verbose:
        print(f"{model.__class__.__name__} pass")


def assert_dist_model_equal(
    model: torch.nn.Module, distributed_model: torch.nn.Module, device_mesh: DeviceMesh, sharding_spec_dict: dict
) -> None:
    state = model.state_dict()
    distributed_state = distributed_model.state_dict()

    assert len(state) == len(distributed_state), f"len {len(state)} vs {len(distributed_state)}"

    for (n1, t1), (n2, t2) in zip(state.items(), distributed_state.items()):
        assert n1 == n2
        t1 = t1.cuda()
        t2 = t2.cuda()
        if n2 in sharding_spec_dict:
            layout = Layout(device_mesh=device_mesh, sharding_spec=sharding_spec_dict[n2], global_shape=t1.shape)
            t2.dist_layout = layout
            t2 = to_global(t2)
        assert torch.equal(t1, t2), f"{n1} {t1} vs {t2}"
