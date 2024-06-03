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
    assert_hf_output_close(org_output, shard_output, ignore_keys=["pred_masks"])

    # do backward
    org_loss.backward()
    shard_loss.backward()

    assert torch.allclose(
        org_loss, shard_loss, atol=1e-5
    ), f"shard model loss is not equal to orgin model loss\n{org_loss}\n{shard_loss}"

    # check grad

    sam = org_model
    sharded_sam = sharded_model

    # check grad
    col_layer_for_check = ["mask_decoder.transformer.layers[0].self_attn.q_proj", "vision_encoder.layers[0].mlp.lin1"]
    row_layer_for_check = ["mask_decoder.transformer.layers[0].self_attn.out_proj", "vision_encoder.layers[0].mlp.lin2"]
    check_grad(sam, sharded_sam, col_layer_for_check, atol=1e-5, rtol=1e-3, dim=0, verbose=False)
    check_grad(sam, sharded_sam, row_layer_for_check, atol=1e-3, rtol=1e-3, dim=1, verbose=False)


@parameterize("enable_fused_normalization", [True, False])
@parameterize("enable_tensor_parallelism", [True, False])
@parameterize("enable_flash_attention", [True, False])
def run_sam_test(enable_fused_normalization, enable_tensor_parallelism, enable_flash_attention):
    sub_model_zoo = model_zoo.get_sub_registry("transformers_sam")
    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        org_model, sharded_model = build_model(
            model_fn, enable_fused_normalization, enable_tensor_parallelism, enable_flash_attention
        )
        check_forward_backward(org_model, sharded_model, data_gen_fn, output_transform_fn, loss_fn)

    torch.cuda.empty_cache()


def check_sam(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_sam_test()


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_sam():
    spawn(check_sam, 2)


if __name__ == "__main__":
    test_sam()
