import pytest
import torch
import torch.distributed as dist
import transformers
from packaging import version

import colossalai
from colossalai.inference.pipeline import PPInferEngine
from colossalai.inference.pipeline.policies import LlamaModelInferPolicy
from colossalai.testing import clear_cache_before_run, parameterize, rerun_if_address_is_in_use, spawn

CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse("11.5")


def data_gen():
    input_ids = torch.tensor([[15496, 11, 616, 3290, 318, 13779, 318, 13779]], dtype=torch.int64)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.int64)
    return dict(input_ids=input_ids, attention_mask=attention_mask)


inputs = data_gen()
for k, v in inputs.items():
    if torch.is_tensor(v) or "Tensor" in v.__class__.__name__:
        new_shape = [1] * v.dim()
        new_shape[0] = 16
        inputs[k] = v.to("cuda").repeat(*new_shape)


def pipeline_inference_test(pp_size, new_length, micro_batch_size):
    model = transformers.LlamaForCausalLM(transformers.LlamaConfig(num_hidden_layers=4))

    engine = PPInferEngine(
        pp_size=pp_size,
        model=model,
        model_policy=LlamaModelInferPolicy(),
        new_length=new_length,
        micro_batch_size=micro_batch_size,
    )
    output = engine.inference(inputs)
    if dist.get_rank() == 0:
        assert len(output[0]) == new_length, f"{len(output)}, {new_length}"


@parameterize("pp_size", [2])
@parameterize("new_length", [4, 8, 16])
@parameterize("micro_batch_size", [1, 4])
@clear_cache_before_run()
def run_pipeline_inference_test(pp_size, new_length, micro_batch_size):
    pipeline_inference_test(pp_size, new_length, micro_batch_size)
    torch.cuda.empty_cache()


def check_pipeline_inference(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_pipeline_inference_test()


@pytest.mark.skipif(not CUDA_SUPPORT, reason="kv-cache manager engine requires cuda version to be higher than 11.5")
@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_pipeline_inference():
    spawn(check_pipeline_inference, nprocs=2)


if __name__ == "__main__":
    test_pipeline_inference()
