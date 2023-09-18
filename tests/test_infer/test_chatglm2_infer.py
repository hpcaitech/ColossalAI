import os

import numpy as np
import pytest
import torch
import torch.distributed as dist
from packaging import version
from transformers import AutoModel, AutoTokenizer

import colossalai
from colossalai.cluster import ProcessGroupMesh
from colossalai.inference.tensor_parallel.engine import TPInferEngine
from colossalai.logging import disable_existing_loggers
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.shardformer.modeling.chatglm2_6b.modeling_chatglm import ChatGLMForConditionalGeneration, ChatGLMModel
from colossalai.testing import clear_cache_before_run, parameterize, rerun_if_address_is_in_use, spawn

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
TPSIZE = 1
BATCH_SIZE = 8
MAX_INPUT_LEN = 12
MAX_OUTPUT_LEN = 100
CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse('11.5')


@parameterize('test_config', [{
    'tp_size': TPSIZE,
}])
def run_chatglm2_test(test_config):

    chatglm2_model_path = "/home/lccjh/data2/lccjh/chatglm2-6b"
    assert os.path.isdir(chatglm2_model_path) is True

    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
    # pad_token_id = 0

    model = ChatGLMForConditionalGeneration.from_pretrained(chatglm2_model_path, pad_token_id=tokenizer.eos_token_id)
    #init_to_get_rotary(model.model, base=10000)
    model = model.half()

    text = ["how is the weather today?", "i am ", "你好？"]
    input_ids = tokenizer.batch_encode_plus(text, return_tensors='pt', padding=True)
    shard_config = ShardConfig(enable_tensor_parallelism=True if test_config['tp_size'] > 1 else False,
                               inference_only=True)
    infer_engine = TPInferEngine(model, shard_config, BATCH_SIZE, MAX_INPUT_LEN, MAX_OUTPUT_LEN)

    generate_kwargs = dict(max_new_tokens=MAX_OUTPUT_LEN, do_sample=False)
    outputs = infer_engine.generate(input_ids, **generate_kwargs)

    # print("outputs.shape: ", outputs[0].shape)
    # print("outputs: ", outputs[0])
    if not dist.is_initialized() or dist.get_rank() == 0:
        for o in outputs:
            output_text = tokenizer.decode(o)
            print(output_text)


def check_chatglm2(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_chatglm2_test()


@pytest.mark.skipif(not CUDA_SUPPORT, reason="kv-cache manager engine requires cuda version to be higher than 11.5")
@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_chatglm2():
    spawn(check_chatglm2, TPSIZE)


if __name__ == "__main__":
    test_chatglm2()
