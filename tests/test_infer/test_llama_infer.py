import os

import pytest
import torch

import colossalai
from colossalai.logging import disable_existing_loggers
from colossalai.testing import clear_cache_before_run, parameterize, rerun_if_address_is_in_use, spawn
from llama_infer_eigine import TPCacheManagerInferenceEngine

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

@parameterize('test_config', [{
    'tp_size': 2,
}])
def run_llama_test(test_config):
    input_len = 1024
    output_len = 128
    bs = 8
    engine = TPCacheManagerInferenceEngine(input_len, output_len, bs, 2)
    engine.generate_data()
    engine.prepare_model()
    engine.init_and_insert_cache_manager()
    
    org_model, sharded_model = engine.build_model()
    engine.model = sharded_model
    
    engine.run_infer()

    torch.cuda.empty_cache()


def check_llama(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_llama_test()


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_llama():
    spawn(check_llama, 2)


if __name__ == "__main__":
    test_llama()
