import os

import pytest
import torch
import numpy as np

import colossalai
from colossalai.logging import disable_existing_loggers
from colossalai.testing import clear_cache_before_run, parameterize, rerun_if_address_is_in_use, spawn
from colossalai.inference.tensor_parallel.llama_infer_engine import TPCacheManagerInferenceEngine

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
TPSIZE = 2

@parameterize('test_config', [{
    'tp_size': TPSIZE,
}])
def run_llama_test(test_config):
    input_len = 1024
    output_len = 128
    bs = 8
    engine = TPCacheManagerInferenceEngine(input_len, output_len, bs, test_config["tp_size"])
    engine.generate_data()
    engine.prepare_model()
    
    engine.build_model()
    
    outputs_list = engine.run_infer(test_origin=False)
    
    torch.cuda.empty_cache()


def check_llama(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_llama_test()


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_llama():
    spawn(check_llama, TPSIZE)


if __name__ == "__main__":
    test_llama()
