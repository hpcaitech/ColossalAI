import os
import warnings

import pytest
import torch
import torch.distributed as dist
from packaging import version
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import BloomForCausalLM, BloomTokenizerFast

import colossalai
from colossalai.inference.tensor_parallel.engine import TPInferEngine
from colossalai.logging import disable_existing_loggers
from colossalai.shardformer import ShardConfig
from colossalai.testing import clear_cache_before_run, parameterize, rerun_if_address_is_in_use, spawn

# MODEL_PATH = "/home/lclcq/share/models--bigscience--bloom-560m/snapshots/4f42c91d806a19ae1a46af6c3fb5f4990d884cd6"
MODEL_PATH = "/home/lclcq/share/llama-7b"

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
TPSIZE = 1
BATCH_SIZE = 4
MAX_INPUT_LEN = 32
MAX_OUTPUT_LEN = 128

CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse('11.5')


@parameterize('test_config', [{
    'tp_size': TPSIZE,
}])
def run_llama_test(test_config):

    model_path = MODEL_PATH
    if os.path.isdir(model_path) is False:
        warnings.warn("Model path does not exist")
        return
    
    # tokenizer = BloomTokenizerFast.from_pretrained(model_path)
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.unk_token_id
    # model = BloomForCausalLM.from_pretrained(model_path, pad_token_id=tokenizer.eos_token_id)
    model = LlamaForCausalLM.from_pretrained(model_path, pad_token_id=tokenizer.eos_token_id)
    model = model.half()

    text = ["Introduce London.", "What is the genus of Poodle?"]
    input_ids = tokenizer.batch_encode_plus(text, return_tensors='pt', padding=True)

    print(input_ids)

    shard_config = ShardConfig(enable_tensor_parallelism=True if test_config['tp_size'] > 1 else False,
                               inference_only=True)
    infer_engine = TPInferEngine(model, shard_config, BATCH_SIZE, MAX_INPUT_LEN, MAX_OUTPUT_LEN)

    generate_kwargs = dict(max_new_tokens=MAX_OUTPUT_LEN, do_sample=False)
    outputs = infer_engine.generate(input_ids, **generate_kwargs)

    assert outputs is not None

    if not dist.is_initialized() or dist.get_rank() == 0:
        for o in outputs:
            output_text = tokenizer.decode(o)
            print(output_text)


def check_llama(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_llama_test()


@pytest.mark.skipif(not CUDA_SUPPORT, reason="kv-cache manager engine requires cuda version to be higher than 11.5")
@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_llama():
    spawn(check_llama, TPSIZE)


if __name__ == "__main__":
    test_llama()
