import tempfile
import pytest
import torch
from torch.optim import Adam
from torchvision.models import resnet18

from colossalai.checkpoint_io import GeneralCheckpointIO
from colossalai.booster.plugin.gemini_plugin import GeminiCheckpointIO
from colossalai.testing import clear_cache_before_run, parameterize

import colossalai
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.utils.cuda import get_current_device
from colossalai.zero import ColoInitContext, ZeroDDP
from colossalai.zero.gemini.chunk import ChunkManager, search_chunk_configuration
from colossalai.zero.gemini.gemini_mgr import GeminiManager
from tests.components_to_test.registry import non_distributed_component_funcs

# ========
# Note:
# 1. due to checkpoint IO can be quite slow if tested with all models, we will only test on resnet for now
# 2. we will test on both sharded and unsharded checkpoints
# 3. implement sharded checkpoint and test it
# ========


@clear_cache_before_run()
@parameterize('use_safetensors', [True, False])
def test_unsharded_checkpoint(use_safetensors: bool):
    # create a model and optimizer
    model = resnet18()
    optimizer = Adam(model.parameters(), lr=0.001)

    # create test data sample
    x = torch.randn(1, 3, 224, 224)

    # run fwd and bwd
    y = model(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()

    # create a temp file for checkpoint
    if use_safetensors:
        suffix = ".safetensors"
    else:
        suffix = ".bin"
    model_ckpt_tempfile = tempfile.NamedTemporaryFile(suffix=suffix)
    optimizer_ckpt_tempfile = tempfile.NamedTemporaryFile()

    # save the model and optimizer
    ckpt_io = GeneralCheckpointIO()
    ckpt_io.save_model(model, model_ckpt_tempfile.name, use_safetensors=use_safetensors)
    ckpt_io.save_optimizer(optimizer, optimizer_ckpt_tempfile.name)

    # create new model
    new_model = resnet18()
    new_optimizer = Adam(new_model.parameters(), lr=0.001)

    # load the model and optimizer
    ckpt_io.load_model(new_model, model_ckpt_tempfile.name)
    ckpt_io.load_optimizer(new_optimizer, optimizer_ckpt_tempfile.name)


    # check for model and optimizer state dict recursively
    recursive_check(model.state_dict(), new_model.state_dict())
    recursive_check(optimizer.state_dict(), new_optimizer.state_dict())

@pytest.mark.parametrize('use_safetensors', [True, False])
def test_sharded_checkpoint(use_safetensors: bool):
    # create a model and optimizer
    model = resnet18()
    optimizer = Adam(model.parameters(), lr=0.001)
    # create test data sample
    x = torch.randn(1, 3, 224, 224)

    # run fwd and bwd
    y = model(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()

    # create a temp file for checkpoint
    if use_safetensors:
        suffix = ".safetensors"
        SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
    else:
        suffix = ".bin"
        WEIGHTS_INDEX_NAME = "model.bin.index.json"
    
    model_ckpt_dir = tempfile.TemporaryDirectory()
    optimizer_ckpt_tempfile = tempfile.NamedTemporaryFile()

    # save the model and optimizer
    ckpt_io = GeneralCheckpointIO()

    ckpt_io.save_model(model, model_ckpt_dir.name, True, True, "", 10, use_safetensors=use_safetensors)
    ckpt_io.save_optimizer(optimizer, optimizer_ckpt_tempfile.name, shard=False)
    
    # create new model
    new_model = resnet18()
    new_optimizer = Adam(new_model.parameters(), lr=0.001)

    ckpt_io.load_model(new_model, str(model_ckpt_dir.name), strict=True)
    ckpt_io.load_optimizer(new_optimizer, optimizer_ckpt_tempfile.name)

    # check for model and optimizer state dict recursively
    recursive_check(model.state_dict(), new_model.state_dict())
    recursive_check(optimizer.state_dict(), new_optimizer.state_dict())

@parameterize('placement_policy', ['cuda', 'cpu'])
@parameterize('model_name', ['bert'])
@parameterize('use_safetensors', [True, False])
def hf_load_colossalai_checkpoint(placement_policy, model_name, use_safetensors: bool):
    from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig, BertForSequenceClassification

    model_ckpt_dir = tempfile.TemporaryDirectory()
    get_components_func = non_distributed_component_funcs.get_callable(model_name)
    model_builder, *_ = get_components_func()

    with ColoInitContext(device=get_current_device()):
        bert_model = model_builder()
    bert_model.config.save_pretrained(save_directory=model_ckpt_dir.name)
    config_dict, *_ = search_chunk_configuration(bert_model, search_range_mb=1, search_interval_byte=100)
    chunk_manager = ChunkManager(config_dict)
    gemini_manager = GeminiManager(placement_policy, chunk_manager)
    bert_model = ZeroDDP(bert_model, gemini_manager)
    bert_model.train()

    ckpt_io = GeminiCheckpointIO()
    if ckpt_io.coordinator.is_master():
        model_size = sum(p.numel() * p.element_size() for p in bert_model.parameters()) / 1024**2
        ckpt_io.save_model(bert_model, model_ckpt_dir.name, True, True, "", (model_size / 3), use_safetensors=use_safetensors)
        new_bert_model = BertForSequenceClassification.from_pretrained(model_ckpt_dir.name)
        recursive_check(bert_model.state_dict(only_rank_0=True, dtype=torch.float32), new_bert_model.state_dict())
    
    model_ckpt_dir.cleanup()
        


@parameterize('placement_policy', ['cuda', 'cpu'])
@parameterize('model_name', ['gpt2', 'bert'])
@parameterize('use_safetensors', [True, False])
def exam_state_dict(placement_policy, model_name: str, use_safetensors: bool):
    get_components_func = non_distributed_component_funcs.get_callable(model_name)
    model_builder, *_ = get_components_func()

    with ColoInitContext(device=get_current_device()):
        model = model_builder()
        new_model = model_builder()

    config_dict, *_ = search_chunk_configuration(model, search_range_mb=1, search_interval_byte=100)
    chunk_manager = ChunkManager(config_dict)
    gemini_manager = GeminiManager(placement_policy, chunk_manager)
    model = ZeroDDP(model, gemini_manager)
    model.train()

    new_config_dict, *_ = search_chunk_configuration(new_model, search_range_mb=1, search_interval_byte=100)
    new_chunk_manager = ChunkManager(new_config_dict)
    new_gemini_manager = GeminiManager(placement_policy, new_chunk_manager)
    new_model = ZeroDDP(new_model, new_gemini_manager)

    model_ckpt_dir = tempfile.TemporaryDirectory()

    ckpt_io = GeminiCheckpointIO()
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    ckpt_io.save_model(model, model_ckpt_dir.name, True, True, "epoch", (model_size / 3), use_safetensors=use_safetensors)

    # load model
    if ckpt_io.coordinator.is_master():
        ckpt_io.load_model(new_model, model_ckpt_dir.name, strict=True)
        model_dict = model.state_dict(only_rank_0=True)
        new_model_dict = new_model.state_dict(only_rank_0=True)
        recursive_check(model_dict, new_model_dict)

    model_ckpt_dir.cleanup()


def run_dist(rank, world_size, port):
    config = {}
    colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    exam_state_dict()
    hf_load_colossalai_checkpoint()


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [4, 4])
@rerun_if_address_is_in_use()
def test_gemini_ckpIO(world_size):
    spawn(run_dist, world_size)


# do recursive check for the optimizer state dict
# if the value is a dict, compare its values
# if the value is a list, comapre all elements one-by-one
# if the value is a torch.Tensor, use torch.equal
# otherwise use assertEqual
def recursive_check(d1, d2):
    for k, v in d1.items():
        if isinstance(v, dict):
            recursive_check(v, d2[k])
        elif isinstance(v, list):
            for i in range(len(v)):
                if isinstance(v[i], torch.Tensor):
                    v[i] = v[i].to("cpu")
                    d2[k][i] = d2[k][i].to("cpu")
                    assert torch.equal(v[i], d2[k][i])
                else:
                    assert v[i] == d2[k][i]
        elif isinstance(v, torch.Tensor):
            v = v.to("cpu")
            d2[k] = d2[k].to("cpu")
            assert torch.equal(v, d2[k])
        else:
            assert v == d2[k]
