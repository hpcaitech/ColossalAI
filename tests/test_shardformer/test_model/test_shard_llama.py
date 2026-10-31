import importlib.util
import os

import pytest
import torch
import torch.distributed as dist
from torch.testing import assert_close

import colossalai
from colossalai._compat import is_transformers_v5
from colossalai.logging import disable_existing_loggers
from colossalai.pipeline.schedule.v_schedule import PipelineGraph
from colossalai.shardformer import PipelineGradientCheckpointConfig
from colossalai.shardformer.layer.utils import Randomizer
from colossalai.tensor.d_tensor.api import clear_layout_converter
from colossalai.testing import clear_cache_before_run, parameterize, rerun_if_address_is_in_use, spawn
from tests.kit.model_zoo import model_zoo
from tests.test_shardformer.test_model._utils import (
    build_model_from_hybrid_plugin,
    check_all_grad_tensors,
    check_loss,
    check_output_hidden_state,
    check_weight,
    get_grad_tensors_for_check,
    run_forward_backward_with_hybrid_plugin,
    unwrap_model,
)

HAS_FLASH_ATTN = importlib.util.find_spec("flash_attn") is not None

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"


@pytest.mark.skipif(not is_transformers_v5(), reason="`create_causal_mask` is only used by the v5 implementation")
def test_llama_pp_all_to_all_causal_mask_uses_full_query_length():
    from transformers.masking_utils import create_causal_mask
    from transformers.models.llama.configuration_llama import LlamaConfig

    from colossalai.shardformer.modeling.llama import _get_causal_mask_input_embeddings

    hidden_states = torch.empty(4, 2, 8)
    assert _get_causal_mask_input_embeddings(hidden_states, query_length=2, sequence_is_sharded=True) is hidden_states
    assert _get_causal_mask_input_embeddings(hidden_states, query_length=4, sequence_is_sharded=False) is hidden_states
    mask_inputs_embeds = _get_causal_mask_input_embeddings(hidden_states, query_length=4, sequence_is_sharded=True)
    assert mask_inputs_embeds.shape == (4, 4, 8)
    assert mask_inputs_embeds.stride(1) == 0
    assert mask_inputs_embeds.untyped_storage().data_ptr() == hidden_states.untyped_storage().data_ptr()

    config = LlamaConfig()
    config._attn_implementation = "eager"
    position_ids = torch.arange(4).unsqueeze(0).expand(4, -1)
    mask = create_causal_mask(
        config=config,
        inputs_embeds=mask_inputs_embeds,
        attention_mask=None,
        past_key_values=None,
        position_ids=position_ids,
        allow_is_causal_skip=False,
    )
    assert mask is not None and mask.shape == (4, 1, 4, 4)


@pytest.mark.skipif(not is_transformers_v5(), reason="`create_causal_mask` is only used by the v5 implementation")
@pytest.mark.parametrize("attn_implementation", ["sdpa", "flash_attention_2", "flex_attention"])
def test_llama_explicit_softmax_mask_is_additive(attn_implementation):
    from transformers.masking_utils import create_causal_mask
    from transformers.models.llama.configuration_llama import LlamaConfig

    from colossalai.shardformer.modeling.llama import _EagerMaskConfig

    # Under these implementations `create_causal_mask` returns a boolean mask, None / a 2D padding
    # mask, or a BlockMask; the explicit-softmax attention needs an additive float mask instead.
    config = LlamaConfig(num_hidden_layers=1, hidden_size=32, intermediate_size=64, num_attention_heads=4)
    config._attn_implementation = attn_implementation
    mask_config = _EagerMaskConfig(config)
    assert mask_config._attn_implementation == "eager"
    assert config._attn_implementation == attn_implementation
    # Every other field is read from the live config, so later changes are seen at once.
    config.num_attention_heads = 8
    assert mask_config.num_attention_heads == 8
    config.num_attention_heads = 4

    future = torch.ones(4, 4, dtype=torch.bool).triu(1)
    for attention_mask in (torch.ones(2, 4, dtype=torch.long), torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])):
        mask = create_causal_mask(
            config=mask_config,
            inputs_embeds=torch.empty(2, 4, 32, dtype=torch.float16),
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=torch.arange(4).unsqueeze(0).expand(2, -1),
            allow_is_causal_skip=False,
        )
        assert mask.dtype == torch.float16 and mask.shape == (2, 1, 4, 4)
        assert (mask[:, :, future] == torch.finfo(torch.float16).min).all()
    assert (mask[0, 0, :, 3] == torch.finfo(torch.float16).min).all()  # the padded key
    assert (mask[1, :, ~future] == 0).all()


def check_forward_backward(model_fn, data_gen_fn, output_transform_fn, loss_fn, test_config):
    enable_gradient_checkpointing = test_config.pop("enable_gradient_checkpointing", False)
    org_model, org_optimizer, sharded_model, sharded_optimizer, criterion, booster = build_model_from_hybrid_plugin(
        model_fn, loss_fn, test_config
    )
    if enable_gradient_checkpointing:
        # org_model.gradient_checkpointing_enable()
        sharded_model.unwrap().gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    org_loss, org_output, sharded_loss, sharded_output = run_forward_backward_with_hybrid_plugin(
        org_model, sharded_model, sharded_optimizer, data_gen_fn, output_transform_fn, criterion, booster
    )

    stage_manager = booster.plugin.stage_manager
    tp_group = booster.plugin.tp_group

    # unwrap model
    llama_model = unwrap_model(org_model, "LlamaModel", "model")
    shard_llama_model = unwrap_model(sharded_model, "LlamaModel", "model")

    row_layer_for_check = ["layers[0].self_attn.q_proj", "embed_tokens"]
    col_layer_for_check = ["layers[0].self_attn.o_proj"]
    # Here we check the grad of layernorm because an all-reduce operation should be performed during sequence parallelism
    norm_layer_for_check = ["layers[0].input_layernorm", "layers[0].post_attention_layernorm"]

    # During pipeline parallelism, we cannot get the grad of norm layer during first stage, so we only check this when pp is not enbaled
    if stage_manager is None:
        norm_layer_for_check.append("norm")

    # Check the grad when using ZeRO-1 and ZeRO-2
    if (
        booster.plugin.zero_stage in [1, 2]
        and booster.plugin.shard_config.enable_sequence_parallelism
        and booster.plugin.shard_config.pipeline_stage_manager is None
        and booster.plugin.shard_config.sequence_parallelism_mode == "all_to_all"
    ):
        master2working = sharded_optimizer.get_master_to_working_map()
        for (name, p1), p2 in zip(
            llama_model.named_parameters(), sharded_optimizer._master_param_groups_of_current_rank[0]
        ):
            working_p = master2working[id(p2)]
            grads = sharded_optimizer.get_partitioned_gradients_by_param_id(0, id(working_p))
            grad_index = (
                0
                if sharded_optimizer._partition_grads
                else sharded_optimizer.pid_to_bucket_store[id(working_p)].local_rank
            )
            grad = grads[grad_index]
            sharded_grad = p1.grad.view(-1).chunk(dist.get_world_size())[dist.get_rank()]
            try:
                assert_close(sharded_grad, grad[: sharded_grad.shape[0]], atol=5e-3, rtol=5e-3, check_dtype=False)
            except Exception as e:
                raise RuntimeError(f"Failed to check grad for {name}") from e

    # Save gradient tensors for comparison between the original model and the sharded model before optimizer step.
    grads_to_check = {}
    if (stage_manager is None or stage_manager.is_first_stage(ignore_chunk=True)) and booster.plugin.zero_stage == 0:
        if test_config["precision"] == "fp32":
            atol, rtol = 1e-6, 1e-4
        else:
            atol, rtol = 5e-3, 5e-3
        row_layer_grads = get_grad_tensors_for_check(
            llama_model, shard_llama_model, row_layer_for_check, tp_group, atol=atol, rtol=rtol, dim=0, verbose=False
        )
        col_layer_grads = get_grad_tensors_for_check(
            llama_model, shard_llama_model, col_layer_for_check, tp_group, atol=atol, rtol=rtol, dim=1, verbose=False
        )
        norm_layer_grads = get_grad_tensors_for_check(
            llama_model,
            shard_llama_model,
            norm_layer_for_check,
            tp_group,
            atol=atol,
            rtol=rtol,
            dim=1,
            verbose=False,
        )
        grads_to_check.update(col_layer_grads)
        grads_to_check.update(row_layer_grads)
        grads_to_check.update(norm_layer_grads)

    # optimizer executes step
    org_optimizer.step()
    sharded_optimizer.step()

    # check last hidden state & loss
    check_flag = False
    if (
        (stage_manager is None)
        or (stage_manager.use_zbv and stage_manager.is_first_stage(ignore_chunk=True))
        or (not stage_manager.use_zbv and stage_manager.is_last_stage(ignore_chunk=True))
    ):
        check_flag = True
    if check_flag:
        if test_config["precision"] == "fp32":
            atol, rtol = 1e-5, 1e-3
        else:
            atol, rtol = 5e-3, 5e-3
        if org_model.__class__.__name__ == "LlamaModel":
            check_output_hidden_state(
                org_output,
                sharded_output,
                stage_manager,
                atol=atol,
                rtol=rtol,
                shard_config=booster.plugin.shard_config,
            )
        check_loss(org_loss, sharded_loss, atol=atol, rtol=rtol)
    # check weights
    if stage_manager is None or stage_manager.is_first_stage(ignore_chunk=True):
        if test_config["precision"] == "fp32":
            atol, rtol = 1e-4, 1e-3
        else:
            atol, rtol = 5e-3, 5e-3
        check_weight(
            llama_model,
            shard_llama_model,
            col_layer_for_check,
            tp_group,
            atol=atol,
            rtol=rtol,
            dim=1,
            verbose=False,
        )

    # check grads
    check_all_grad_tensors(grads_to_check)
    torch.cuda.empty_cache()


@parameterize(
    "test_config",
    [
        # Double Ring Attention
        {
            "tp_size": 1,
            "pp_size": 1,
            "sp_size": 4,
            "num_microbatches": 1,
            "enable_sequence_parallelism": True,
            "sequence_parallelism_mode": "ring_attn",
            "use_lazy_init": True,
            "zero_stage": 0,
            "precision": "fp16",
            "initial_scale": 1,
        },
        # Ring Attention + PP
        {
            "tp_size": 1,
            "pp_size": 2,
            "sp_size": 2,
            "num_microbatches": 2,
            "enable_sequence_parallelism": True,
            "sequence_parallelism_mode": "ring_attn",
            "use_lazy_init": True,
            "zero_stage": 1,
            "precision": "fp16",
            "initial_scale": 1,
        },
        # Ring Attention + TP
        {
            "tp_size": 2,
            "pp_size": 1,
            "sp_size": 2,
            "num_microbatches": 1,
            "enable_sequence_parallelism": True,
            "sequence_parallelism_mode": "ring_attn",
            "use_lazy_init": True,
            "zero_stage": 2,
            "precision": "fp16",
            "initial_scale": 1,
        },
        {  # Ulysess + TP
            "tp_size": 2,
            "pp_size": 1,
            "sp_size": 2,
            "num_microbatches": 1,
            "enable_sequence_parallelism": True,
            "sequence_parallelism_mode": "all_to_all",
            "enable_all_optimization": True,
            "use_lazy_init": True,
            "zero_stage": 0,
            "precision": "fp16",
            "initial_scale": 1,
        },
        {  # Ulysess + PP
            "tp_size": 1,
            "pp_size": 2,
            "sp_size": 2,
            "num_microbatches": 2,
            "enable_sequence_parallelism": True,
            "sequence_parallelism_mode": "all_to_all",
            "enable_all_optimization": True,
            "use_lazy_init": True,
            "zero_stage": 1,
            "precision": "fp16",
            "initial_scale": 1,
        },
        # SP + PP without flash attention: later stages build the eager causal mask from a sequence shard.
        # v5 only: the v4 implementation (`_llama_tf4`, upstream as-is) builds that mask from the shard
        # length and fails there with a 12 vs 24 size mismatch, leaving the other stage blocked in P2P.
        *(
            [
                {  # Ulysess + PP
                    "tp_size": 1,
                    "pp_size": 2,
                    "sp_size": 2,
                    "num_microbatches": 2,
                    "enable_sequence_parallelism": True,
                    "sequence_parallelism_mode": "all_to_all",
                    "use_lazy_init": False,
                    "zero_stage": 1,
                    "precision": "fp16",
                    "initial_scale": 1,
                },
                {  # split_gather + PP
                    "tp_size": 2,
                    "pp_size": 2,
                    "num_microbatches": 2,
                    "enable_sequence_parallelism": True,
                    "sequence_parallelism_mode": "split_gather",
                    "use_lazy_init": False,
                    "zero_stage": 0,
                    "precision": "fp32",
                    "initial_scale": 1,
                },
                {  # ring + PP; zero_stage 0 so that the weight gradients are checked
                    "tp_size": 2,
                    "pp_size": 2,
                    "num_microbatches": 2,
                    "enable_sequence_parallelism": True,
                    "sequence_parallelism_mode": "ring",
                    "use_lazy_init": False,
                    "zero_stage": 0,
                    "precision": "fp32",
                    "initial_scale": 1,
                },
            ]
            if is_transformers_v5()
            else []
        ),
        {
            "tp_size": 2,
            "pp_size": 1,
            "sp_size": 1,
            "num_microbatches": 1,
            "enable_sequence_parallelism": True,
            "sequence_parallelism_mode": "ring",
            "enable_flash_attention": True,
            "use_lazy_init": True,
            "zero_stage": 2,
            "precision": "fp16",
            "initial_scale": 1,
        },
        {
            "tp_size": 2,
            "pp_size": 2,
            "num_microbatches": 2,
            "enable_all_optimization": True,
            "use_lazy_init": True,
            "precision": "fp16",
            "initial_scale": 1,
            "enable_gradient_checkpointing": True,
            "gradient_checkpoint_config": PipelineGradientCheckpointConfig(gradient_checkpointing_ratio=0.5),
        },
        {
            "tp_size": 1,
            "pp_size": 2,
            "num_microbatches": 4,
            "use_lazy_init": False,
            "precision": "fp32",
            "enable_gradient_checkpointing": True,
            "gradient_checkpoint_config": PipelineGradientCheckpointConfig(num_ckpt_layers_per_stage=[4, 0]),
        },
        {
            "tp_size": 2,
            "pp_size": 1,
            "enable_all_optimization": True,
            "use_lazy_init": True,
            "zero_stage": 2,
            "precision": "fp16",
            "initial_scale": 1,
        },
        {
            "tp_size": 1,
            "pp_size": 2,
            "num_microbatches": 2,
            "enable_all_optimization": True,
            "use_lazy_init": True,
            "zero_stage": 1,
            "precision": "fp16",
            "initial_scale": 1,
        },
    ],
)
def run_llama_test(test_config):
    if test_config.get("sequence_parallelism_mode") == "ring_attn" and not HAS_FLASH_ATTN:
        if torch.distributed.get_rank() == 0:
            print("Skipping Llama ring-attention config because flash-attn is not installed")
        return

    sub_model_zoo = model_zoo.get_sub_registry("transformers_llama")
    if test_config.get("pp_style", None) == "zbv":
        mem_f = 34 * 32 + 5 * 4 * 16
        mem_w = -32 * 32
        mem_b = -mem_w - mem_f
        scheduler_nodes = PipelineGraph(
            n_stage=test_config["pp_size"],
            n_micro=test_config["num_microbatches"],
            f_cost=1000,
            b_cost=1000,
            w_cost=1000,
            c_cost=1,
            f_mem=mem_f,
            b_mem=mem_b,
            w_mem=mem_w,
        ).get_v_schedule()
        test_config["scheduler_nodes"] = scheduler_nodes
    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        if test_config.get("sequence_parallelism_mode", None) == "ring_attn" and "causal" not in name:
            continue
        try:
            check_forward_backward(model_fn, data_gen_fn, output_transform_fn, loss_fn, test_config)
        except Exception as e:
            print(f"Failed config: {test_config}, model name: {name}")
            raise e

    clear_layout_converter()
    Randomizer.reset_index()
    torch.cuda.empty_cache()


@parameterize(
    "test_config",
    [
        {
            "tp_size": 2,
            "pp_size": 2,
            "num_microbatches": 4,
            "enable_all_optimization": False,
            "use_lazy_init": False,
            "precision": "fp32",
            "initial_scale": 1,
        },
        {
            "tp_size": 2,
            "pp_size": 2,
            "num_microbatches": 4,
            "enable_all_optimization": False,
            "use_lazy_init": False,
            "precision": "fp16",
            "zero_stage": 1,
            "initial_scale": 1,
        },
        {
            "tp_size": 2,
            "pp_size": 2,
            "pp_style": "interleaved",
            "num_model_chunks": 2,
            "num_microbatches": 4,
            "enable_all_optimization": False,
            "precision": "fp16",
            "zero_stage": 1,
            "initial_scale": 1,
            "enable_gradient_checkpointing": True,
            "gradient_checkpoint_config": PipelineGradientCheckpointConfig(
                num_ckpt_layers_per_stage=[0, 1, 2, 2],
            ),
        },
    ],
)
def run_llama_3d_test(test_config):
    sub_model_zoo = model_zoo.get_sub_registry("transformers_llama")

    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        try:
            check_forward_backward(model_fn, data_gen_fn, output_transform_fn, loss_fn, test_config)
        except Exception as e:
            print(f"Failed config: {test_config}")
            raise e

    clear_layout_converter()
    Randomizer.reset_index()
    torch.cuda.empty_cache()


def check_llama(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_llama_test()


def check_llama_3d(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_llama_3d_test()


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_llama():
    spawn(check_llama, 4)


@pytest.mark.largedist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_llama_3d():
    spawn(check_llama_3d, 8)


if __name__ == "__main__":
    test_llama()
    test_llama_3d()
