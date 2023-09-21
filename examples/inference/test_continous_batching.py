import os

import pytest
import torch
from packaging import version
from vllm import LLM

import colossalai
from colossalai.inference.tensor_parallel.engine import TPInferEngine
from colossalai.logging import disable_existing_loggers
from colossalai.testing import clear_cache_before_run, parameterize, rerun_if_address_is_in_use, spawn

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
TPSIZE = 1
MAX_OUTPUT_LEN = 100

CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse('11.5')


@parameterize('test_config', [{
    'tp_size': TPSIZE,
}])
def run_llama_test(test_config):
    model = '/data/scratch/llama-7b-hf'
    tokenizer = 'hf-internal-testing/llama-tokenizer'

    test_prompts = [
        "A robot may not injure a human being",
        "To be or not to be,",
        "What is the meaning of life?",
        "It is only with the heart that one can see rightly",
        "Can you introduce Beijing?",
    ]

    infer_engine = TPInferEngine(
        model=model,
        max_output_len=MAX_OUTPUT_LEN,
        use_continous_batching=True,
        tokenizer=tokenizer,
    )
    generate_kwargs = dict(do_sample=False)
    outputs = infer_engine.generate(test_prompts, **generate_kwargs)

    print("outputs: ", outputs)


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
