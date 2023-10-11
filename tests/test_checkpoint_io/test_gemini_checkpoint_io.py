import os

import pytest
import torch
import torch.distributed as dist
from transformers import LlamaForCausalLM
from utils import shared_tempdir

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin
from colossalai.lazy import LazyInitContext
from colossalai.nn.optimizer import HybridAdam
from colossalai.testing import (
    check_state_dict_equal,
    clear_cache_before_run,
    parameterize,
    rerun_if_address_is_in_use,
    spawn,
)
from tests.kit.model_zoo import model_zoo

MODEL_PLACEMENT_CONFIGS = [
    {"placement_policy": "static", "shard_param_frac": 0.0},  # zero2
    {"placement_policy": "static", "shard_param_frac": 1.0},  # zero3
    {"placement_policy": "static", "shard_param_frac": 0.5},  # zero3-half
]

OPTIM_PLACEMENT_CONFIGS = [
    {"placement_policy": "static", "shard_param_frac": 0.0, "offload_optim_frac": 0.0},  # zero2
    {"placement_policy": "static", "shard_param_frac": 0.0, "offload_optim_frac": 1.0},  # zero2-offload
    {"placement_policy": "static", "shard_param_frac": 0.0, "offload_optim_frac": 0.5},  # zero2-offload-half
]


@clear_cache_before_run()
@parameterize("placement_config", MODEL_PLACEMENT_CONFIGS)
@parameterize("model_name", ["transformers_bert_for_sequence_classification"])
@parameterize("use_safetensors", [False, True])
def exam_state_dict_with_origin(placement_config, model_name, use_safetensors: bool):
    from transformers import BertForSequenceClassification

    (model_fn, data_gen_fn, output_transform_fn, _, _) = next(iter(model_zoo.get_sub_registry(model_name).values()))
    bert_model = model_fn()

    with shared_tempdir() as tempdir:
        pretrained_path = os.path.join(tempdir, "pretrained")
        bert_model.config.save_pretrained(save_directory=pretrained_path)

        plugin = GeminiPlugin(**placement_config)
        booster = Booster(plugin=plugin)
        bert_model, _, _, _, _ = booster.boost(bert_model)
        model_size = sum(p.numel() * p.element_size() for p in bert_model.parameters()) / 1024**2

        booster.save_model(
            bert_model, pretrained_path, True, True, "", (model_size / 3), use_safetensors=use_safetensors
        )
        dist.barrier()

        new_bert_model = BertForSequenceClassification.from_pretrained(pretrained_path)
        check_state_dict_equal(bert_model.state_dict(only_rank_0=False), new_bert_model.state_dict(), False)


@clear_cache_before_run()
@parameterize("placement_config", OPTIM_PLACEMENT_CONFIGS)
@parameterize("shard", [False, True])
@parameterize("model_name", ["transformers_gpt"])
@parameterize("size_per_shard", [32])
def exam_state_dict(placement_config, shard: bool, model_name: str, size_per_shard: int):
    (model_fn, data_gen_fn, output_transform_fn, _, _) = next(iter(model_zoo.get_sub_registry(model_name).values()))
    criterion = lambda x: x.mean()
    plugin = GeminiPlugin(**placement_config, precision="fp16", initial_scale=(2**14))
    booster = Booster(plugin=plugin)

    model = model_fn()
    new_model = model_fn()
    optimizer = HybridAdam(model.parameters(), lr=0.001)
    model, optimizer, criterion, _, _ = booster.boost(model, optimizer, criterion)
    new_optimizer = HybridAdam(new_model.parameters(), lr=0.001)
    new_model, new_optimizer, criterion, _, _ = booster.boost(new_model, new_optimizer, criterion)

    data = data_gen_fn()
    data = {k: v.to("cuda") if torch.is_tensor(v) or "Tensor" in v.__class__.__name__ else v for k, v in data.items()}
    output = model(**data)
    output = output_transform_fn(output)
    output_key = list(output.keys())[0]
    loss = criterion(output[output_key])

    booster.backward(loss, optimizer)
    optimizer.step()

    with shared_tempdir() as tempdir:
        model_ckpt_path = f"{tempdir}/model"
        optimizer_ckpt_path = f"{tempdir}/optimizer"
        booster.save_model(model, model_ckpt_path, shard=shard, size_per_shard=size_per_shard)

        booster.save_optimizer(optimizer, optimizer_ckpt_path, shard=shard, size_per_shard=size_per_shard)
        dist.barrier()

        booster.load_model(new_model, model_ckpt_path)
        check_state_dict_equal(
            model.state_dict(only_rank_0=False), new_model.state_dict(only_rank_0=False), False, ignore_dtype=True
        )

        booster.load_optimizer(new_optimizer, optimizer_ckpt_path)
        check_state_dict_equal(
            optimizer.state_dict(only_rank_0=False), new_optimizer.state_dict(only_rank_0=False), False
        )

        # Check the new model/optimizer can successfully run.
        data = data_gen_fn()
        data = {
            k: v.to("cuda") if torch.is_tensor(v) or "Tensor" in v.__class__.__name__ else v for k, v in data.items()
        }
        output = new_model(**data)
        output = output_transform_fn(output)
        output_key = list(output.keys())[0]
        loss = criterion(output[output_key])
        booster.backward(loss, new_optimizer)
        new_optimizer.step()
        booster.save_model(new_model, model_ckpt_path, shard=shard)
        booster.save_optimizer(new_optimizer, optimizer_ckpt_path, shard=shard)


def exam_lazy_from_pretrained():
    llama_path = os.environ["LLAMA_PATH"]
    plugin = GeminiPlugin()
    booster = Booster(plugin=plugin)
    orig_model = LlamaForCausalLM.from_pretrained(llama_path)
    orig_state_dict = {k: v.half() for k, v in orig_model.state_dict().items()}
    with LazyInitContext():
        model = LlamaForCausalLM.from_pretrained(llama_path)
    model, *_ = booster.boost(model)
    with shared_tempdir() as tempdir:
        save_path = os.path.join(tempdir, "model.pt")
        booster.save_model(model, save_path, shard=False)
        dist.barrier()
        state_dict = torch.load(save_path, map_location="cpu")
        check_state_dict_equal(state_dict, orig_state_dict, False, ignore_dtype=True)


def run_dist(rank, world_size, port):
    config = {}
    colossalai.launch(config=config, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    exam_state_dict()
    exam_state_dict_with_origin()
    exam_lazy_from_pretrained()


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [2])
@rerun_if_address_is_in_use()
def test_gemini_ckpIO(world_size):
    spawn(run_dist, world_size)
