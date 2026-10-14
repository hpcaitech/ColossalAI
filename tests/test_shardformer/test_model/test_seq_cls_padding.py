import pytest
import torch
from transformers import (
    LlamaConfig,
    LlamaForSequenceClassification,
    OPTConfig,
    OPTForSequenceClassification,
    Qwen2Config,
    Qwen2ForSequenceClassification,
)

import colossalai
from colossalai.cluster import ProcessGroupMesh
from colossalai.logging import disable_existing_loggers
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer import ShardConfig
from colossalai.shardformer.modeling.llama import LlamaPipelineForwards
from colossalai.shardformer.modeling.opt import OPTPipelineForwards
from colossalai.shardformer.modeling.qwen2 import Qwen2PipelineForwards
from colossalai.testing import clear_cache_before_run, rerun_if_address_is_in_use, spawn

NUM_LAYERS = 2
COMMON_CONFIG = dict(
    vocab_size=64, hidden_size=32, num_hidden_layers=NUM_LAYERS, num_attention_heads=4, pad_token_id=0, num_labels=3
)

MODELS = {
    "llama": (
        LlamaConfig(intermediate_size=64, num_key_value_heads=4, **COMMON_CONFIG),
        LlamaForSequenceClassification,
        LlamaPipelineForwards.llama_for_sequence_classification_forward,
    ),
    "qwen2": (
        Qwen2Config(intermediate_size=64, num_key_value_heads=4, **COMMON_CONFIG),
        Qwen2ForSequenceClassification,
        Qwen2PipelineForwards.qwen2_for_sequence_classification_forward,
    ),
    "opt": (
        OPTConfig(ffn_dim=64, word_embed_proj_dim=32, max_position_embeddings=32, **COMMON_CONFIG),
        OPTForSequenceClassification,
        OPTPipelineForwards.opt_for_sequence_classification_forward,
    ),
}

# The first row of each batch is padded, the second one is not.
INPUT_IDS = {
    "left": [[0, 0, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14]],
    "right": [[5, 6, 7, 8, 0, 0], [9, 10, 11, 12, 13, 14]],
}


def run_seq_cls_padding():
    device = torch.device("cuda")
    stage_manager = PipelineStageManager(ProcessGroupMesh(1), pipeline_axis=0)
    assert stage_manager.is_first_stage() and stage_manager.is_last_stage()
    shard_config = ShardConfig(enable_tensor_parallelism=False, pipeline_stage_manager=stage_manager)

    for name, (config, model_cls, pipeline_forward) in MODELS.items():
        torch.manual_seed(0)
        model = model_cls(config).to(device).eval()
        for padding_side, ids in INPUT_IDS.items():
            input_ids = torch.tensor(ids, device=device)
            attention_mask = (input_ids != config.pad_token_id).long()
            with torch.no_grad():
                expected = model(input_ids=input_ids, attention_mask=attention_mask).logits
                output = pipeline_forward(
                    model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    stage_manager=stage_manager,
                    stage_index=[0, NUM_LAYERS],
                    shard_config=shard_config,
                ).logits
            assert torch.allclose(output, expected, atol=1e-5), f"{name} with {padding_side} padding"


def check_seq_cls_padding(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_seq_cls_padding()


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_seq_cls_padding():
    spawn(check_seq_cls_padding, 1)


if __name__ == "__main__":
    test_seq_cls_padding()
