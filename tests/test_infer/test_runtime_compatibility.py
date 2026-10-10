"""Small regressions for interfaces exercised by the distributed CI suites."""

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding
from transformers.models.mixtral.modeling_mixtral import MixtralAttention, MixtralConfig, MixtralRotaryEmbedding

from colossalai.booster.plugin import HybridParallelPlugin
from colossalai.inference.spec.drafter import Drafter
from colossalai.shardformer.modeling.llama import LlamaPipelineForwards, get_llama_flash_attention_forward
from colossalai.shardformer.modeling.mixtral import get_mixtral_flash_attention_forward
from colossalai.shardformer.shard import ShardConfig
from colossalai.testing import rerun_on_exception


def test_retry_preserves_pytest_marks():
    @pytest.mark.skipif(True, reason="optional dependency unavailable")
    def original():
        pass

    wrapped = rerun_on_exception()(original)
    assert wrapped.pytestmark == original.pytestmark
    assert wrapped.__wrapped__ is original


def test_plugin_cleanup_does_not_hide_rank_error():
    mesh = Mock()
    plugin = object.__new__(HybridParallelPlugin)
    plugin.pg_mesh = mesh
    try:
        raise RuntimeError("rank-local forward failure")
    except RuntimeError:
        del plugin
    mesh.destroy_mesh_process_groups.assert_not_called()
    plugin = object.__new__(HybridParallelPlugin)
    plugin.pg_mesh = mesh
    del plugin
    mesh.destroy_mesh_process_groups.assert_called_once()


def test_drafter_resumes_legacy_cache():
    torch.manual_seed(123)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    drafter = Drafter(
        LlamaForCausalLM(config),
        SimpleNamespace(eos_token_id=-1),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    inputs = torch.tensor([[5, 6, 7]])
    first = drafter.speculate(inputs, 2)
    resumed = drafter.speculate(first.next_tokens[-1:].unsqueeze(0), 2, first.past_key_values)
    combined = drafter.speculate(inputs, 4)
    torch.testing.assert_close(resumed.next_tokens, combined.next_tokens[2:])
    torch.testing.assert_close(resumed.logits, combined.logits[2:])
    assert isinstance(resumed.past_key_values, tuple)
    assert resumed.past_key_values[0][0].shape[2] == inputs.shape[1] + 3


@pytest.mark.parametrize("kv_heads", [2, 4])
def test_mixtral_sdpa_attention_layout(kv_heads):
    config = MixtralConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=kv_heads,
        num_hidden_layers=1,
    )
    config._attn_implementation = "sdpa"
    attention = MixtralAttention(config, layer_idx=0).eval()
    attention.num_heads = config.num_attention_heads
    attention.num_key_value_heads = config.num_key_value_heads
    hidden = torch.randn(2, 5, config.hidden_size)
    positions = torch.arange(5).unsqueeze(0)
    embeddings = MixtralRotaryEmbedding(config)(hidden, positions)
    expected, _ = attention(hidden, embeddings, attention_mask=None)
    forward = get_mixtral_flash_attention_forward(ShardConfig(enable_tensor_parallelism=False))
    actual, _ = forward(attention, hidden, attention_mask=None, position_embeddings=embeddings)
    torch.testing.assert_close(actual, expected)


def test_llama_attention_slices_reserved_cache_mask_column():
    config = LlamaConfig(hidden_size=32, num_attention_heads=4, num_key_value_heads=2)
    config._attn_implementation = "eager"
    attention = LlamaAttention(config, layer_idx=0).eval()
    attention.num_heads = config.num_attention_heads
    hidden = torch.randn(2, 5, config.hidden_size)
    positions = torch.arange(5).unsqueeze(0)
    embeddings = LlamaRotaryEmbedding(config)(hidden, positions)
    mask = torch.zeros(2, 1, 5, 6).masked_fill(
        torch.ones(5, 6, dtype=torch.bool).triu(1), torch.finfo(hidden.dtype).min
    )
    expected, _ = attention(hidden, embeddings, attention_mask=mask)
    forward = get_llama_flash_attention_forward(ShardConfig(enable_tensor_parallelism=False))
    actual, _ = forward(attention, hidden, position_embeddings=embeddings, attention_mask=mask)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("mode", ["split_gather", "ring", "all_to_all"])
def test_llama_pipeline_mask_uses_full_sequence(mode, monkeypatch):
    from colossalai.shardformer.modeling import llama as module

    model = LlamaModel(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
    )
    masks = []

    def decoder(hidden, **kwargs):
        masks.append(kwargs["attention_mask"])
        return (hidden,)

    monkeypatch.setattr(model.layers[0], "forward", decoder)
    monkeypatch.setattr(module, "gather_sp_output", lambda hidden, config: hidden)
    stage = SimpleNamespace(is_first_stage=lambda: False, is_last_stage=lambda: True)
    shard = SimpleNamespace(
        sequence_parallel_size=2,
        sequence_parallel_process_group=None,
        sequence_parallelism_mode=mode,
        enable_flash_attention=False,
        parallel_output=False,
    )
    LlamaPipelineForwards.llama_model_forward(
        model,
        hidden_states=torch.randn(2, 4, 32),
        stage_manager=stage,
        stage_index=[0, 1],
        shard_config=shard,
        use_cache=False,
    )
    assert masks[0].shape == (2, 1, 8, 8)
    assert masks[0][0, 0, 0, 1] < -1e30
    assert masks[0][0, 0, 7, 0] == 0


@pytest.mark.parametrize("family", ["qwen2", "qwen3"])
@pytest.mark.parametrize("mode", ["split_gather", "ring", "all_to_all"])
def test_qwen_pipeline_mask_uses_full_sequence(family, mode, monkeypatch):
    transformers_module = importlib.import_module(f"transformers.models.{family}.modeling_{family}")
    module = importlib.import_module(f"colossalai.shardformer.modeling.{family}")
    prefix = "Qwen2" if family == "qwen2" else "Qwen3"
    config = getattr(transformers_module, prefix + "Config")(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    model = getattr(transformers_module, prefix + "Model")(config)
    shapes = []

    def prepare(shape, *args, **kwargs):
        shapes.append(shape)
        return {}

    monkeypatch.setattr(module.ColoAttention, "prepare_attn_kwargs", prepare)
    monkeypatch.setattr(module, "gather_sp_output", lambda hidden, config: hidden)
    monkeypatch.setattr(model.layers[0], "forward", lambda hidden, *args: (hidden,))
    stage = SimpleNamespace(is_first_stage=lambda: False, is_last_stage=lambda: True)
    shard = SimpleNamespace(
        sequence_parallel_size=2,
        sequence_parallel_process_group=None,
        sequence_parallelism_mode=mode,
        enable_sequence_parallelism=True,
        enable_flash_attention=True,
        parallel_output=False,
    )
    forwards = getattr(module, prefix + "PipelineForwards")
    forward = getattr(forwards, family + "_model_forward")
    forward(
        model,
        hidden_states=torch.randn(2, 4, 32),
        stage_manager=stage,
        stage_index=[0, 1],
        shard_config=shard,
    )
    assert shapes == [(2, 1, 8, 8)]
