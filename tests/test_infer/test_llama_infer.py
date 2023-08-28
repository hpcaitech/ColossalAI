import os

import pytest
import torch
from torch import distributed as dist

import colossalai
from colossalai.logging import disable_existing_loggers
from colossalai.shardformer.layer.utils import Randomizer
from colossalai.tensor.d_tensor.api import clear_layout_converter
from colossalai.testing import clear_cache_before_run, parameterize, rerun_if_address_is_in_use, spawn
from tests.kit.model_zoo import model_zoo
from tests.test_infer._utils import build_model, run_infer

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


def check_infer(model_fn, data_gen_fn, output_transform_fn, test_config):
    org_model, sharded_model = build_model(model_fn, **test_config)

    org_output, infer_output = run_infer(org_model, sharded_model, data_gen_fn, output_transform_fn)

    print('original output', org_output[0])
    print('infer output', infer_output[0])


@parameterize('test_config', [{
    'enable_flash_attention': False,
}])
def run_llama_test(test_config):

    sub_model_zoo = model_zoo.get_sub_registry('transformers_llama')

    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        if name != "transformers_llama":
            continue
        check_infer(model_fn, data_gen_fn, output_transform_fn, test_config)
    torch.cuda.empty_cache()


def check_llama(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_llama_test()


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_llama():
    spawn(check_llama, 1)


if __name__ == "__main__":
    test_llama()
