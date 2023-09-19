import os

import numpy as np
import pytest
import torch
import torch.distributed as dist
from packaging import version
from transformers import AutoModel, AutoTokenizer

import colossalai
from colossalai.inference.tensor_parallel.engine import TPInferEngine
from colossalai.logging import disable_existing_loggers
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.shardformer.modeling.chatglm2_6b.modeling_chatglm import ChatGLMForConditionalGeneration, ChatGLMModel
from colossalai.testing import clear_cache_before_run, parameterize, rerun_if_address_is_in_use, spawn
from tests.kit.model_zoo import model_zoo

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

    sub_model_zoo = model_zoo.get_sub_registry('transformers_chatglm_for_conditional_generation')

    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
    # pad_token_id = 0
    for name, (model_fn, data_gen_fn, _, _, _) in sub_model_zoo.items():
        orig_model = model_fn()
        orig_model = orig_model.half()
        text = ["how is the weather today?"]
        input_ids = tokenizer.batch_encode_plus(text, return_tensors='pt', padding=True)
        shard_config = ShardConfig(enable_tensor_parallelism=True if test_config['tp_size'] > 1 else False,
                                   inference_only=True)
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
