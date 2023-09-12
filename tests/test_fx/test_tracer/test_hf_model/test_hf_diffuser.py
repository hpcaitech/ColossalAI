import pytest
import torch

from colossalai.fx import symbolic_trace
from colossalai.testing import clear_cache_before_run
from colossalai.testing.random import seed_all
from tests.kit.model_zoo import model_zoo


def assert_dict(da, db, assert_fn):
    assert len(da) == len(db)
    for k, v in da.items():
        assert k in db
        if not torch.is_tensor(v):
            continue
        u = db.get(k)
        assert_fn(u, v)


def trace_and_compare(model_cls, data, output_fn):
    model = model_cls()
    model.eval()

    concrete_args = {k: v for k, v in data.items() if not torch.is_tensor(v)}
    meta_args = {k: v.to('meta') for k, v in data.items() if torch.is_tensor(v)}
    gm = symbolic_trace(model, concrete_args=concrete_args, meta_args=meta_args)

    # run forward
    with torch.no_grad():
        fx_out = gm(**data)
        non_fx_out = model(**data)

    # compare output
    transformed_fx_out = output_fn(fx_out)
    transformed_non_fx_out = output_fn(non_fx_out)

    def assert_fn(ta, tb):
        assert torch.equal(ta, tb)

    assert_dict(transformed_fx_out, transformed_non_fx_out, assert_fn)


@pytest.mark.skip(reason='cannot pass this test yet')
@clear_cache_before_run()
def test_diffusers():
    seed_all(9091, cuda_deterministic=True)

    sub_model_zoo = model_zoo.get_sub_registry('diffusers')

    for name, (model_fn, data_gen_fn, output_transform_fn, _, _, attribute) in sub_model_zoo.items():
        data = data_gen_fn()
        trace_and_compare(model_fn, data, output_transform_fn)
        torch.cuda.synchronize()
        print(f"{name:40s} √")


@clear_cache_before_run()
def test_torch_diffusers():
    seed_all(65535, cuda_deterministic=True)

    sub_model_zoo = model_zoo.get_sub_registry('diffusers')

    for name, (model_fn, data_gen_fn, output_transform_fn, _, attribute) in sub_model_zoo.items():
        data = data_gen_fn()
        model = model_fn()
        output = model(**data)
        torch.cuda.synchronize()
        print(f"{name:40s} √")


if __name__ == "__main__":
    test_torch_diffusers()
