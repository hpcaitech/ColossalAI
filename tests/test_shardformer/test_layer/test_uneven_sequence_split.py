import torch
import torch.distributed as dist
from torch import nn
from transformers import LlamaConfig, LlamaModel

from colossalai.shardformer.layer._operation import (
    gather_forward_split_backward,
    pad_sequence_parallel_inputs,
    split_forward_gather_backward,
)
from colossalai.shardformer.layer.loss import dist_cross_entropy
from colossalai.shardformer.modeling.llama import LlamaPipelineForwards, get_llama_flash_attention_forward
from colossalai.testing import rerun_if_address_is_in_use, spawn


def _check_uneven_sequence_split(rank, world_size, port):
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=f"tcp://127.0.0.1:{port}")
    try:
        # Five tokens cannot be evenly split over two sequence-parallel ranks.
        input_ = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
        input_.requires_grad_()

        local = split_forward_gather_backward(input_, dim=1, process_group=dist.group.WORLD)
        assert local.shape == (2, 3, 3)

        gathered = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(gathered, local)
        gathered = torch.cat(gathered, dim=1)
        expected = torch.cat((input_.detach(), torch.zeros(2, 1, 3)), dim=1)
        torch.testing.assert_close(gathered, expected)

        # A downstream operation may touch padded positions.  Those positions
        # must not change the shape or values of the input gradient.  Rank 1's
        # final element is padding, while rank 0's corresponding element is a
        # real token, so use different weights to distinguish the two.
        weight = torch.ones_like(local)
        weight[:, -1] = 10 + rank
        (local * weight).sum().backward()
        expected_grad = torch.ones_like(input_)
        expected_grad[:, 2] = 10
        torch.testing.assert_close(input_.grad, expected_grad)

        # Keep the model-side metadata on the same padded extent as hidden
        # states, then trim the gathered result back to the logical length.
        hidden = torch.randn(2, 5, 3, requires_grad=True)
        attention_mask = torch.ones(2, 5, dtype=torch.long)
        position_ids = torch.arange(5, dtype=torch.long).unsqueeze(0)
        cache_position = torch.arange(5, dtype=torch.long)
        hidden, attention_mask, position_ids, cache_position = pad_sequence_parallel_inputs(
            hidden, attention_mask, position_ids, cache_position, target_length=6
        )
        hidden.retain_grad()
        assert hidden.shape == (2, 6, 3)
        assert attention_mask.shape == (2, 6)
        assert attention_mask[:, -1].eq(0).all()
        assert position_ids.shape == (1, 6)
        assert cache_position.tolist() == [0, 1, 2, 3, 4, 5]

        local_hidden = split_forward_gather_backward(hidden, dim=1, process_group=dist.group.WORLD)
        logical_hidden = gather_forward_split_backward(
            local_hidden, dim=1, process_group=dist.group.WORLD, output_dim_size=5
        )
        assert logical_hidden.shape == (2, 5, 3)
        logical_hidden.sum().backward()
        assert hidden.grad is not None and hidden.grad.shape == hidden.shape
        assert hidden.grad[:, -1].eq(0).all()

        # The language-model loss consumes the local padded logits and the
        # original (uneven) labels.  Ignored targets must mask the synthetic
        # tail while the reduced loss still match the unsharded reference.
        vocab_size = 7
        global_logits = (
            torch.arange(2 * 5 * vocab_size, dtype=torch.float32).reshape(2, 5, vocab_size) / vocab_size
        ).requires_grad_()
        local_logits = split_forward_gather_backward(global_logits, dim=1, process_group=dist.group.WORLD)
        loss_config = type("ShardConfig", (), {})()
        loss_config.sequence_parallel_process_group = dist.group.WORLD
        loss_config.sequence_parallel_size = world_size
        loss_config.sequence_parallelism_mode = "all_to_all"
        loss_config.parallel_output = True
        loss_config.enable_tensor_parallelism = False
        labels = torch.tensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
        loss = dist_cross_entropy(labels, local_logits, loss_config, vocab_size, global_logits.dtype)
        reference = nn.functional.cross_entropy(
            global_logits.detach()[:, :-1].reshape(-1, vocab_size),
            labels[:, 1:].reshape(-1),
            reduction="mean",
        )
        torch.testing.assert_close(loss, reference)
        loss.backward()
        assert global_logits.grad is not None and global_logits.grad.shape == global_logits.shape
        assert global_logits.grad[:, -1].abs().sum() == 0

        # Exercise the Llama model path: all sequence-side metadata must use
        # the padded extent while the public output keeps the original length.
        llama_config = LlamaConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
        llama_config._attn_implementation = "eager"
        llama = LlamaModel(llama_config)
        attention = llama.layers[0].self_attn
        # Model-parallel projections expose a head-sharded shape to the
        # all-to-all path, matching the dimensions used after tensor sharding.
        attention.num_heads = 1
        attention.num_key_value_heads = 1
        attention.num_key_value_groups = 1
        attention.head_dim = 4
        attention.q_proj = nn.Linear(8, 8, bias=False)
        attention.k_proj = nn.Linear(8, 8, bias=False)
        attention.v_proj = nn.Linear(8, 8, bias=False)
        attention.o_proj = nn.Linear(8, 8, bias=False)

        shard_config = type("ShardConfig", (), {})()
        shard_config.sequence_parallelism_mode = "all_to_all"
        shard_config.sequence_parallel_process_group = dist.group.WORLD
        shard_config.sequence_parallel_size = world_size
        shard_config.enable_sequence_parallelism = True
        shard_config.enable_flash_attention = False
        shard_config.fp8_communication = False
        shard_config.parallel_output = False
        shard_config.gradient_checkpoint_config = None
        attention.forward = get_llama_flash_attention_forward(
            shard_config,
            sp_mode="all_to_all",
            sp_size=world_size,
            sp_group=dist.group.WORLD,
        ).__get__(attention, type(attention))

        model_output = LlamaPipelineForwards.llama_model_forward(
            llama,
            input_ids=torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]),
            attention_mask=torch.ones(2, 5, dtype=torch.long),
            shard_config=shard_config,
            return_dict=True,
        )
        assert model_output.last_hidden_state.shape == (2, 5, 8)
        model_output.last_hidden_state.square().mean().backward()
    finally:
        dist.destroy_process_group()


@rerun_if_address_is_in_use()
def test_uneven_sequence_split_cpu():
    spawn(_check_uneven_sequence_split, nprocs=2)
