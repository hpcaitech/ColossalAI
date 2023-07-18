import os

import pytest
import torch
import torch.distributed as dist
from utils import shared_tempdir

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin
from colossalai.nn.optimizer import HybridAdam
from colossalai.testing import (
    check_state_dict_equal,
    clear_cache_before_run,
    parameterize,
    rerun_if_address_is_in_use,
    spawn,
)
from tests.kit.model_zoo import model_zoo


@clear_cache_before_run()
@parameterize('placement_policy', ['cuda', 'cpu'])
@parameterize('model_name', ['transformers_bert_for_sequence_classification'])
@parameterize('use_safetensors', [False, True])
def exam_state_dict_with_origin(placement_policy, model_name, use_safetensors: bool):
    from transformers import BertForSequenceClassification
    (model_fn, data_gen_fn, output_transform_fn, _, _) = next(iter(model_zoo.get_sub_registry(model_name).values()))
    bert_model = model_fn()

    with shared_tempdir() as tempdir:
        pretrained_path = os.path.join(tempdir, 'pretrained')
        bert_model.config.save_pretrained(save_directory=pretrained_path)

        plugin = GeminiPlugin(placement_policy=placement_policy)
        booster = Booster(plugin=plugin)
        bert_model, _, _, _, _ = booster.boost(bert_model)
        model_size = sum(p.numel() * p.element_size() for p in bert_model.parameters()) / 1024**2

        booster.save_model(bert_model,
                           pretrained_path,
                           True,
                           True,
                           '', (model_size / 3),
                           use_safetensors=use_safetensors)
        dist.barrier()

        new_bert_model = BertForSequenceClassification.from_pretrained(pretrained_path)
        check_state_dict_equal(bert_model.unwrap().state_dict(only_rank_0=False, dtype=torch.float32),
                               new_bert_model.state_dict(), False)


@clear_cache_before_run()
@parameterize('placement_policy', ['cuda', 'cpu'])
@parameterize('shard', [False])
@parameterize('model_name', ['transformers_gpt'])
@parameterize('size_per_shard', [32])
def exam_state_dict(placement_policy, shard: bool, model_name: str, size_per_shard: int):
    (model_fn, data_gen_fn, output_transform_fn, _, _) = next(iter(model_zoo.get_sub_registry(model_name).values()))
    criterion = lambda x: x.mean()
    plugin = GeminiPlugin(placement_policy=placement_policy, precision="fp16", initial_scale=(2**14))
    booster = Booster(plugin=plugin)

    model = model_fn()
    new_model = model_fn()
    optimizer = HybridAdam(model.parameters(), lr=0.001)
    model, optimizer, criterion, _, _ = booster.boost(model, optimizer, criterion)
    new_optimizer = HybridAdam(new_model.parameters(), lr=0.001)
    new_model, new_optimizer, criterion, _, _ = booster.boost(new_model, new_optimizer, criterion)

    data = data_gen_fn()
    data = {k: v.to('cuda') if torch.is_tensor(v) or 'Tensor' in v.__class__.__name__ else v for k, v in data.items()}
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
        check_state_dict_equal(model.unwrap().state_dict(only_rank_0=False),
                               new_model.unwrap().state_dict(only_rank_0=False), False)

        booster.load_optimizer(new_optimizer, optimizer_ckpt_path)
        check_state_dict_equal(optimizer.unwrap().state_dict(only_rank_0=False),
                               new_optimizer.unwrap().state_dict(only_rank_0=False), False)

        # Check the new model/optimizer can successfully run.
        data = data_gen_fn()
        data = {
            k: v.to('cuda') if torch.is_tensor(v) or 'Tensor' in v.__class__.__name__ else v for k, v in data.items()
        }
        output = new_model(**data)
        output = output_transform_fn(output)
        output_key = list(output.keys())[0]
        loss = criterion(output[output_key])
        booster.backward(loss, new_optimizer)
        new_optimizer.step()
        booster.save_model(new_model, model_ckpt_path, shard=shard)
        booster.save_optimizer(new_optimizer, optimizer_ckpt_path, shard=shard)


def run_dist(rank, world_size, port):
    config = {}
    colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    exam_state_dict()
    exam_state_dict_with_origin()


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 2])
@rerun_if_address_is_in_use()
def test_gemini_ckpIO(world_size):
    spawn(run_dist, world_size)
