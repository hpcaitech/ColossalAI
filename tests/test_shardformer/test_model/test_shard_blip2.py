import pytest
import torch

import colossalai
from colossalai.logging import disable_existing_loggers
from colossalai.testing import (
    assert_hf_output_close,
    clear_cache_before_run,
    parameterize,
    rerun_if_address_is_in_use,
    spawn,
)
from tests.kit.model_zoo import model_zoo
from tests.test_shardformer.test_model._utils import build_model, check_grad, run_forward


def check_forward_backward(org_model, sharded_model, data_gen_fn, output_transform_fn, loss_fn):
    # check forward
    org_output, org_loss, shard_output, shard_loss = run_forward(
        org_model, sharded_model, data_gen_fn, output_transform_fn, loss_fn
    )
    assert_hf_output_close(org_output, shard_output, ignore_keys=["past_key_values"])

    # do backward
    org_loss.backward()
    shard_loss.backward()

    assert torch.allclose(
        org_loss, shard_loss, atol=1e-5
    ), f"shard model loss is not equal to orgin model loss\n{org_loss}\n{shard_loss}"

    # check grad

    blip2 = org_model
    sharded_blip2 = sharded_model

    # check grad
    col_layer_for_check = [
        "vision_model.encoder.layers[0].self_attn.qkv",
        "qformer.encoder.layer[0].attention.attention.query",
        "language_model.model.decoder.layers[0].self_attn.k_proj",
    ]
    row_layer_for_check = [
        "vision_model.encoder.layers[0].self_attn.projection",
        "qformer.encoder.layer[0].attention.output.dense",
        "language_model.model.decoder.layers[0].self_attn.out_proj",
    ]
    check_grad(
        blip2,
        sharded_blip2,
        col_layer_for_check,
        atol=1e-6,
        rtol=1e-5,
        dim=0,
        verbose=False,
    )
    check_grad(
        blip2,
        sharded_blip2,
        row_layer_for_check,
        atol=1e-6,
        rtol=1e-5,
        dim=1,
        verbose=False,
    )


@parameterize("enable_fused_normalization", [True, False])
@parameterize("enable_tensor_parallelism", [True, False])
@parameterize("enable_flash_attention", [True])
@parameterize("enable_jit_fused", [True])
def run_blip2_test(
    enable_fused_normalization,
    enable_tensor_parallelism,
    enable_flash_attention,
    enable_jit_fused,
):
    sub_model_zoo = model_zoo.get_sub_registry("transformers_blip2")
    for name, (
        model_fn,
        data_gen_fn,
        output_transform_fn,
        loss_fn,
        _,
    ) in sub_model_zoo.items():
        org_model, sharded_model = build_model(
            model_fn,
            enable_fused_normalization,
            enable_tensor_parallelism,
            enable_flash_attention,
            enable_jit_fused,
            dtype=torch.float,
        )
        check_forward_backward(org_model, sharded_model, data_gen_fn, output_transform_fn, loss_fn)

    torch.cuda.empty_cache()


def check_blip2(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(
        rank=rank,
        world_size=world_size,
        host="localhost",
        port=port,
        backend="nccl",
    )
    run_blip2_test()


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_blip2():
    spawn(check_blip2, 2)


if __name__ == "__main__":
    test_blip2()
