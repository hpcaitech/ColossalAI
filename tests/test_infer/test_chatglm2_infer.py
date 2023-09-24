import os

import pytest
import torch
import torch.distributed as dist
from packaging import version
from transformers import AutoTokenizer

import colossalai
from colossalai.inference.tensor_parallel.engine import TPInferEngine
from colossalai.logging import disable_existing_loggers
from colossalai.shardformer import ShardConfig
from colossalai.shardformer.modeling.chatglm2_6b.modeling_chatglm import ChatGLMForConditionalGeneration
from colossalai.testing import clear_cache_before_run, parameterize, rerun_if_address_is_in_use, spawn
from tests.kit.model_zoo.transformers.chatglm2 import infer_config

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
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
    # pad_token_id = 0
    model_fn = lambda: ChatGLMForConditionalGeneration(infer_config, empty_init=False)
    orig_model = model_fn()
    orig_model = orig_model.half()
    text = ["how is the weather today?"]
    input_ids = tokenizer.batch_encode_plus(text, return_tensors="pt", padding=True)
    shard_config = ShardConfig(
        enable_tensor_parallelism=True if test_config["tp_size"] > 1 else False, inference_only=True
    )
    infer_engine = TPInferEngine(orig_model, shard_config, BATCH_SIZE, MAX_INPUT_LEN, MAX_OUTPUT_LEN)

    generate_kwargs = dict(max_new_tokens=MAX_OUTPUT_LEN, do_sample=False)
    outputs = infer_engine.generate(input_ids, **generate_kwargs)
    assert outputs is not None

    # print("outputs.shape: ", outputs[0].shape)
    # print("outputs: ", outputs[0])
    if not dist.is_initialized() or dist.get_rank() == 0:
        for o in outputs:
            output_text = tokenizer.decode(o)
            print(output_text)


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
