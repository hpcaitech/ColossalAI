import os
import shutil

import pytest
import torch
import torch.distributed as dist
from colossal_moe.models.mixtral_checkpoint import MixtralMoECheckpointIO
from colossal_moe.models.mixtral_layer import replace_moe_layer
from colossal_moe.models.mixtral_policy import MixtralForCausalLMPolicy
from transformers.models.mixtral import MixtralConfig, MixtralForCausalLM

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin.moe_hybrid_parallel_plugin import MoeHybridParallelPlugin
from colossalai.moe.manager import MOE_MANAGER
from colossalai.testing import DummyDataloader, check_state_dict_equal, rerun_if_address_is_in_use, spawn
from colossalai.utils import get_current_device


def data_gen_fn(batch_size: int = 2, max_length: int = 4, vocab_size: int = 20):
    input_ids = torch.randint(0, vocab_size, (batch_size, max_length), device=get_current_device())
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids,
    }


def run_fwd_bwd(
    model, data, label, criterion, optimizer, enable_autocast=False, pipeline=False, booster=None, plugin=None
):
    model.train()
    if pipeline:
        train_dataloader_iter = DummyDataloader(data_gen_fn, length=1)
        is_pp_last_stage = booster.plugin.stage_manager.is_last_stage()
        y = booster.execute_pipeline(
            train_dataloader_iter,
            model,
            lambda x, y: x.loss,
            optimizer,
            return_loss=True,
            return_outputs=True,
        )
        # Backward and optimize
        if is_pp_last_stage:
            loss = y["loss"]
    else:
        if criterion:
            y = model(data).logits
            loss = criterion(y)
        else:
            loss = model(data, label)
        loss = loss.float()

        if optimizer is not None:
            optimizer.backward(loss)
        else:
            loss.backward()
    return y


def get_config():
    config = MixtralConfig(
        vocab_size=300,
        hidden_size=32,
        intermediate_size=16,
        num_hidden_layers=2,
        dropout_rate=0.0,
    )
    return config


def get_model(parallel):
    config = get_config()
    model = MixtralForCausalLM(config).to(torch.bfloat16)
    replace_moe_layer(model)
    optim = torch.optim.Adam(model.parameters())
    args = dict(
        precision="bf16",
        tp_size=1,
        zero_stage=1,
        custom_policy=MixtralForCausalLMPolicy(),
        checkpoint_io=MixtralMoECheckpointIO,
    )
    if parallel == "ep":
        plugin = MoeHybridParallelPlugin(
            pp_size=1,
            **args,
        )
    elif parallel == "hybrid":
        plugin = MoeHybridParallelPlugin(
            pp_size=2,
            microbatch_size=1,
            **args,
        )
    booster = Booster(plugin=plugin)
    model, optim, _, _, _ = booster.boost(model=model, optimizer=optim)
    return model, booster, optim


def _test_moe_checkpoint(parallel):
    if dist.get_rank() == 0:
        if os.path.exists("./tmp_ckpt1"):
            shutil.rmtree("./tmp_ckpt1")
        if os.path.exists("./tmp_ckpt2"):
            shutil.rmtree("./tmp_ckpt2")
    dist.barrier()

    if parallel == None:
        MOE_MANAGER.setup(
            parallel=None,
        )
    elif parallel == "ep":
        MOE_MANAGER.setup(
            parallel="EP",
        )
    elif parallel == "hybrid":
        MOE_MANAGER.setup(
            parallel="EP",
            mode="fixed",
            fixed_dp_size=1,
            fixed_ep_size=2,
            fixed_pp_size=2,
        )
    model1, booster1, optim1 = get_model(parallel)
    model2, booster2, optim2 = get_model(parallel)
    # param ckpt
    # check not equal
    try:
        check_state_dict_equal(model1.state_dict(), model2.state_dict(), False)
        raise AssertionError("state_dict should not be equal")
    except:
        pass
    # shard
    booster1.save_model(model1, "./tmp_ckpt1", shard=True, size_per_shard=1)
    booster2.load_model(model2, "./tmp_ckpt1")
    # check
    check_state_dict_equal(model1.state_dict(), model2.state_dict(), False)

    # optim ckpt
    criterion = lambda x: x.mean()
    data = torch.randint(0, 4, (2, 4)).cuda()
    label = torch.randint(0, 4, (2,)).cuda()
    if parallel == "hybrid":
        kwargs = {"pipeline": True, "booster": booster1, "plugin": booster1.plugin}
    else:
        kwargs = {}
    run_fwd_bwd(model1, data, label, criterion, optim1, **kwargs)
    optim1.step()
    optim1.zero_grad()
    # shard
    booster1.save_optimizer(optim1, "./tmp_ckpt2", shard=True, size_per_shard=1)
    dist.barrier()
    booster2.load_optimizer(optim2, "./tmp_ckpt2")
    # check
    check_state_dict_equal(optim1.optim.state_dict(), optim2.optim.state_dict(), False)

    if dist.get_rank() == 0:
        shutil.rmtree("./tmp_ckpt1")
        shutil.rmtree("./tmp_ckpt2")


def _run_dist(rank, world_size, port, parallel):
    colossalai.launch(
        config=dict(),
        rank=rank,
        world_size=world_size,
        host="localhost",
        port=port,
        backend="nccl",
    )
    _test_moe_checkpoint(parallel)


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [4])
@pytest.mark.parametrize("parallel", ["ep", "hybrid"])
@rerun_if_address_is_in_use()
def test_moe_checkpoint(world_size, parallel):
    spawn(_run_dist, world_size, parallel=parallel)


if __name__ == "__main__":
    test_moe_checkpoint(world_size=4, parallel="hybrid")
