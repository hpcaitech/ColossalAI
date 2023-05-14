import tempfile

import pytest
import torch

import colossalai
from colossalai.booster.plugin.gemini_plugin import GeminiCheckpointIO
from colossalai.testing import check_state_dict_equal, parameterize, rerun_if_address_is_in_use, spawn
from colossalai.utils.cuda import get_current_device
from colossalai.zero import ColoInitContext, ZeroDDP
from colossalai.zero.gemini.chunk import ChunkManager, search_chunk_configuration
from colossalai.zero.gemini.gemini_mgr import GeminiManager
from tests.components_to_test.registry import non_distributed_component_funcs


@parameterize('placement_policy', ['cuda', 'cpu'])
@parameterize('model_name', ['bert'])
@parameterize('use_safetensors', [True, False])
def exam_state_dict_with_origin(placement_policy, model_name, use_safetensors: bool):
    from transformers import BertForSequenceClassification

    model_ckpt_dir = tempfile.TemporaryDirectory()
    get_components_func = non_distributed_component_funcs.get_callable(model_name)
    model_builder, *_ = get_components_func()
    with ColoInitContext(device=(get_current_device())):
        bert_model = model_builder()
    bert_model.config.save_pretrained(save_directory=(model_ckpt_dir.name))

    config_dict, *_ = search_chunk_configuration(bert_model, search_range_mb=1, search_interval_byte=100)
    chunk_manager = ChunkManager(config_dict)
    gemini_manager = GeminiManager(placement_policy, chunk_manager)
    bert_model = ZeroDDP(bert_model, gemini_manager)
    bert_model.train()

    ckpt_io = GeminiCheckpointIO()
    if ckpt_io.coordinator.is_master():
        model_size = sum(p.numel() * p.element_size() for p in bert_model.parameters()) / 1024**2
        ckpt_io.save_model(bert_model, (model_ckpt_dir.name),
                           True,
                           True,
                           '', (model_size / 3),
                           use_safetensors=use_safetensors)
        new_bert_model = BertForSequenceClassification.from_pretrained(model_ckpt_dir.name)
        check_state_dict_equal(bert_model.state_dict(only_rank_0=True, dtype=(torch.float32)),
                               new_bert_model.state_dict(), False)
    model_ckpt_dir.cleanup()


@parameterize('placement_policy', ['cuda', 'cpu'])
@parameterize('model_name', ['gpt2', 'bert'])
@parameterize('use_safetensors', [True, False])
def exam_state_dict(placement_policy, model_name: str, use_safetensors: bool):
    get_components_func = non_distributed_component_funcs.get_callable(model_name)
    model_builder, *_ = get_components_func()
    with ColoInitContext(device=(get_current_device())):
        model = model_builder()
        new_model = model_builder()
    config_dict, *_ = search_chunk_configuration(model, search_range_mb=1, search_interval_byte=100)
    chunk_manager = ChunkManager(config_dict)
    gemini_manager = GeminiManager(placement_policy, chunk_manager)
    model = ZeroDDP(model, gemini_manager)

    model.train()
    #new model
    new_config_dict, *_ = search_chunk_configuration(new_model, search_range_mb=1, search_interval_byte=100)
    new_chunk_manager = ChunkManager(new_config_dict)
    new_gemini_manager = GeminiManager(placement_policy, new_chunk_manager)
    new_model = ZeroDDP(new_model, new_gemini_manager)

    model_ckpt_dir = tempfile.TemporaryDirectory()
    ckpt_io = GeminiCheckpointIO()
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    ckpt_io.save_model(model, (model_ckpt_dir.name),
                       True,
                       True,
                       'epoch', (model_size / 3),
                       use_safetensors=use_safetensors)

    if ckpt_io.coordinator.is_master():
        ckpt_io.load_model(new_model, (model_ckpt_dir.name), strict=True)
        model_dict = model.state_dict(only_rank_0=True)
        new_model_dict = new_model.state_dict(only_rank_0=True)
        check_state_dict_equal(model_dict, new_model_dict, False)
    model_ckpt_dir.cleanup()


def run_dist(rank, world_size, port):
    config = {}
    colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    exam_state_dict()
    exam_state_dict_with_origin()


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [4, 4])
@rerun_if_address_is_in_use()
def test_gemini_ckpIO(world_size):
    spawn(run_dist, world_size)
