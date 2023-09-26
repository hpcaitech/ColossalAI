import pytest
import torch
from packaging import version

import colossalai
from colossalai.inference.tensor_parallel import TPInferEngine
from colossalai.logging import disable_existing_loggers
from colossalai.shardformer import ShardConfig
from colossalai.testing import clear_cache_before_run, parameterize, rerun_if_address_is_in_use, spawn
from tests.kit.model_zoo import model_zoo

TP_SIZE = 2
MAX_BATCH_SIZE = 4
MAX_INPUT_LEN = 16
MAX_OUTPUT_LEN = 32

CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse("11.5")


@parameterize(
    "test_config",
    [
        {
            "tp_size": TP_SIZE,
        }
    ],
)
def run(test_config):
    sub_model_zoo = model_zoo.get_sub_registry("transformers_bloom_for_causal_lm")
    for name, (model_fn, data_gen_fn, _, _, _) in sub_model_zoo.items():
        orig_model = model_fn()
        orig_model = orig_model.half()
        data = data_gen_fn()

        shard_config = ShardConfig(
            enable_tensor_parallelism=True if test_config["tp_size"] > 1 else False, inference_only=True
        )
        infer_engine = TPInferEngine(orig_model, shard_config, MAX_BATCH_SIZE, MAX_INPUT_LEN, MAX_OUTPUT_LEN)

        generate_kwargs = dict(do_sample=False)
        outputs = infer_engine.generate(data, **generate_kwargs)

        assert outputs is not None


def check_bloom(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run()


@pytest.mark.skipif(not CUDA_SUPPORT, reason="kv-cache manager engine requires cuda version to be higher than 11.5")
@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_bloom_infer():
    spawn(check_bloom, TP_SIZE)


if __name__ == "__main__":
    test_bloom_infer()
