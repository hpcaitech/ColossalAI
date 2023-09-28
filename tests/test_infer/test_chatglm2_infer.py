import os

import pytest
import torch
from packaging import version

import colossalai
from colossalai.inference.tensor_parallel.engine import TPInferEngine
from colossalai.logging import disable_existing_loggers
from colossalai.shardformer import ShardConfig
from colossalai.shardformer.modeling.chatglm2_6b.configuration_chatglm import ChatGLMConfig
from colossalai.shardformer.modeling.chatglm2_6b.modeling_chatglm import ChatGLMForConditionalGeneration
from colossalai.testing import clear_cache_before_run, parameterize, rerun_if_address_is_in_use, spawn

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
TPSIZE = 1
BATCH_SIZE = 8
MAX_INPUT_LEN = 12
MAX_OUTPUT_LEN = 100
CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse("11.5")


@parameterize(
    "test_config",
    [
        {
            "tp_size": TPSIZE,
        }
    ],
)
def run_chatglm2_test(test_config):
    chatglm_config = ChatGLMConfig(
        num_layers=2,
        vocab_size=1200,
        use_cache=True,
        multi_query_attention=True,
        multi_query_group_num=2,
        num_attention_heads=8,
        hidden_size=1024,
    )
    model = ChatGLMForConditionalGeneration(chatglm_config)
    model = model.half()

    shard_config = ShardConfig(
        enable_tensor_parallelism=True if test_config["tp_size"] > 1 else False, inference_only=True
    )
    infer_engine = TPInferEngine(model, shard_config, BATCH_SIZE, MAX_INPUT_LEN, MAX_OUTPUT_LEN)
    generate_kwargs = dict(max_new_tokens=MAX_OUTPUT_LEN, do_sample=False)

    input_tokens = {
        "input_ids": torch.randint(1, 1000, (BATCH_SIZE, MAX_INPUT_LEN), device="cuda"),
        "attention_mask": torch.ones((BATCH_SIZE, MAX_INPUT_LEN), device="cuda"),
    }
    outputs = infer_engine.generate(input_tokens, **generate_kwargs)

    assert outputs is not None


def check_chatglm2(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_chatglm2_test()


@pytest.mark.skipif(not CUDA_SUPPORT, reason="kv-cache manager engine requires cuda version to be higher than 11.5")
@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_chatglm2():
    spawn(check_chatglm2, TPSIZE)


if __name__ == "__main__":
    test_chatglm2()
