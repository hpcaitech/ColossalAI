import copy
import faulthandler

import pytest
import torch
import torch.distributed as dist

import colossalai
from colossalai._compat import is_transformers_v5
from colossalai.accelerator import set_accelerator
from colossalai.booster.plugin import MoeHybridParallelPlugin
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from tests.kit.model_zoo import model_zoo
from tests.test_checkpoint_io.utils import shared_tempdir
from tests.test_shardformer.test_model._utils import (
    build_model_from_hybrid_plugin,
    run_forward_backward_with_hybrid_plugin,
)


def _tolerances(precision):
    return (1e-4, 1e-4) if precision == "fp32" else (2e-2, 2e-2)


def _reference_grad(name, reference, module):
    if reference is None:
        return None
    if ".mlp.experts." in name:
        local = module.local_expert_range
        return reference[local.start : local.stop]
    return reference


def _check_parameters(reference_model, actual_model, actual_optimizer, zero_stage, precision):
    reference_params = dict(reference_model.named_parameters())
    actual_modules = dict(actual_model.unwrap().named_modules())
    atol, rtol = _tolerances(precision)
    errors = []
    for name, param in actual_model.unwrap().named_parameters():
        reference_param = reference_params[name]
        owner_name = name.rsplit(".", 2)[0]
        owner = actual_modules.get(owner_name)
        expected = _reference_grad(name, reference_param.grad, owner)
        if zero_stage == 0:
            actual = param.grad
        else:
            actual = actual_optimizer.get_param_grad(param)
        if expected is None:
            if actual is not None and torch.count_nonzero(actual) != 0:
                errors.append(f"{name}: expected no gradient")
            continue
        if actual is None:
            errors.append(f"{name}: missing gradient")
            continue
        try:
            torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)
        except AssertionError as exc:
            errors.append(f"{name}: {exc}")
    assert not errors, "\n".join(errors)


def _check_weights(reference_model, actual_model, precision):
    reference_params = dict(reference_model.named_parameters())
    actual_modules = dict(actual_model.unwrap().named_modules())
    atol, rtol = _tolerances(precision)
    for name, param in actual_model.unwrap().named_parameters():
        expected = reference_params[name].detach()
        owner = actual_modules.get(name.rsplit(".", 2)[0])
        if ".mlp.experts." in name:
            local = owner.local_expert_range
            expected = expected[local.start : local.stop]
        torch.testing.assert_close(param.detach().float(), expected.float(), atol=atol, rtol=rtol)


def _check_checkpoint(model_fn, reference_model, model, optimizer, booster, precision):
    with shared_tempdir() as tempdir:
        checkpoint = f"{tempdir}/model"
        booster.save_model(model, checkpoint, shard=True, use_safetensors=True)
        dist.barrier()

        restored_hf = type(reference_model).from_pretrained(checkpoint).cuda()
        if precision != "fp32":
            restored_hf = restored_hf.to(torch.bfloat16)
        reference_params = dict(reference_model.named_parameters())
        for name, param in restored_hf.named_parameters():
            torch.testing.assert_close(param.float(), reference_params[name].float(), atol=2e-2, rtol=2e-2)

        restored = model_fn().cuda()
        if precision != "fp32":
            restored = restored.to(torch.bfloat16)
        restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)
        restored, restored_optimizer, _, _, _ = booster.boost(restored, restored_optimizer)
        booster.load_model(restored, checkpoint, strict=True)
        saved_params = dict(model.named_parameters())
        for name, param in restored.named_parameters():
            torch.testing.assert_close(param, saved_params[name], atol=0, rtol=0)
        dist.barrier()


def check_forward_backward(model_fn, data_gen_fn, output_transform_fn, loss_fn, test_config):
    enable_gradient_checkpointing = test_config.pop("enable_gradient_checkpointing", False)
    org_model, org_optimizer, sharded_model, sharded_optimizer, criterion, booster = build_model_from_hybrid_plugin(
        model_fn, loss_fn, test_config, pluggin_cls=MoeHybridParallelPlugin
    )
    if enable_gradient_checkpointing:
        kwargs = {"use_reentrant": False}
        org_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)
        sharded_model.unwrap().gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)

    precision = booster.plugin.precision
    if precision == "bf16":
        org_model = org_model.to(torch.bfloat16)
    org_loss, org_output, sharded_loss, sharded_output = run_forward_backward_with_hybrid_plugin(
        org_model, sharded_model, sharded_optimizer, data_gen_fn, output_transform_fn, criterion, booster
    )

    atol, rtol = _tolerances(precision)
    torch.testing.assert_close(sharded_loss.float(), org_loss.float(), atol=atol, rtol=rtol)
    output_key = "last_hidden_state" if hasattr(org_output, "last_hidden_state") else "logits"
    torch.testing.assert_close(
        getattr(sharded_output, output_key).float(), getattr(org_output, output_key).float(), atol=atol, rtol=rtol
    )
    _check_weights(org_model, sharded_model, precision)
    _check_parameters(org_model, sharded_model, sharded_optimizer, booster.plugin.zero_stage, precision)

    org_optimizer.step()
    sharded_optimizer.step()
    _check_weights(org_model, sharded_model, precision)
    _check_checkpoint(model_fn, org_model, sharded_model, sharded_optimizer, booster, precision)

    dist.barrier()
    booster.plugin.pg_mesh.destroy_mesh_process_groups()
    dist.barrier()
    torch.cuda.empty_cache()


@parameterize(
    "test_config",
    [
        {"tp_size": 1, "pp_size": 1, "ep_size": 1, "zero_stage": 0, "precision": "fp32"},
        {
            "tp_size": 1,
            "pp_size": 1,
            "ep_size": 1,
            "zero_stage": 1,
            "precision": "bf16",
            "use_lazy_init": True,
            "enable_gradient_checkpointing": True,
            "overlap_communication": False,
        },
        {
            "tp_size": 1,
            "pp_size": 1,
            "ep_size": 1,
            "zero_stage": 2,
            "precision": "bf16",
            "overlap_communication": False,
        },
        {
            "tp_size": 1,
            "pp_size": 1,
            "ep_size": 2,
            "zero_stage": 1,
            "precision": "bf16",
            "overlap_communication": False,
        },
        {
            "tp_size": 1,
            "pp_size": 1,
            "ep_size": 2,
            "zero_stage": 2,
            "precision": "bf16",
            "use_lazy_init": True,
            "overlap_communication": False,
        },
    ],
)
def run_qwen3_moe_test(test_config):
    entries = model_zoo.get_sub_registry("transformers_qwen3_moe")
    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in entries.items():
        print(f"START model={name}, config={test_config}", flush=True)
        try:
            check_forward_backward(model_fn, data_gen_fn, output_transform_fn, loss_fn, copy.deepcopy(test_config))
        except Exception:
            print(f"Failed model={name}, config={test_config}", flush=True)
            raise
        print(f"END model={name}, config={test_config}", flush=True)


def _run_dist(rank, world_size, port):
    set_accelerator("cuda")
    torch.cuda.set_device(rank)
    torch.empty(1, device=f"cuda:{rank}")
    faulthandler.dump_traceback_later(120, repeat=True)
    try:
        colossalai.launch(
            rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl", local_rank=rank
        )
        run_qwen3_moe_test()
    finally:
        faulthandler.cancel_dump_traceback_later()


@pytest.mark.dist
@pytest.mark.skipif(not is_transformers_v5(), reason="Qwen3-MoE sharding needs transformers v5")
@rerun_if_address_is_in_use()
def test_qwen3_moe():
    spawn(_run_dist, 4)


if __name__ == "__main__":
    test_qwen3_moe()
