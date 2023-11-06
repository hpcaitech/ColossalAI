import importlib
import os
import shutil
import sys

import pytest
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from transformers.models.llama import LlamaConfig

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin.moe_hybrid_parallel_plugin import MoeHybridParallelPlugin
from colossalai.moe.manager import MOE_MANAGER
from colossalai.testing import check_state_dict_equal, rerun_if_address_is_in_use, spawn
from colossalai.utils import get_current_device

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "examples/language/openmoe",
    )
)

OpenMoeForCausalLM = importlib.import_module("model.modeling_openmoe").OpenMoeForCausalLM
set_openmoe_args = importlib.import_module("model.modeling_openmoe").set_openmoe_args
OpenMoeForCausalLMPolicy = importlib.import_module("model.openmoe_policy").OpenMoeForCausalLMPolicy


class RandomDataset(Dataset):
    def __init__(self, num_samples: int = 4, max_length: int = 16, vocab_size: int = 10, tokenizer=None):
        self.num_samples = num_samples
        self.max_length = max_length
        self.input_ids = torch.randint(0, vocab_size, (num_samples, max_length), device=get_current_device())
        self.attention_mask = torch.ones_like(self.input_ids)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.input_ids[idx],
        }


def run_fwd_bwd(
    model, data, label, criterion, optimizer, enable_autocast=False, pipeline=False, booster=None, plugin=None
):
    model.train()
    with torch.cuda.amp.autocast(enabled=enable_autocast):
        if pipeline:
            dataset = RandomDataset(num_samples=20)
            collate_fn = None
            dataloader = plugin.prepare_dataloader(
                dataset, batch_size=1, shuffle=True, drop_last=True, collate_fn=collate_fn
            )
            train_dataloader_iter = iter(dataloader)
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
    config = LlamaConfig(
        vocab_size=300,
        hidden_size=8,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        head_dim=4,
        dropout_rate=0.0,
        hidden_act="swiglu",
    )
    set_openmoe_args(config, num_experts=8, moe_layer_interval=1)
    return config


def get_model(parallel):
    config = get_config()
    model = OpenMoeForCausalLM(config)
    optim = torch.optim.Adam(model.parameters())

    if parallel == None:
        plugin = MoeHybridParallelPlugin(
            tp_size=1,
            pp_size=1,
            zero_stage=2,
            custom_policy=OpenMoeForCausalLMPolicy(),
        )
    elif parallel == "ep":
        plugin = MoeHybridParallelPlugin(
            tp_size=1,
            pp_size=1,
            zero_stage=2,
            custom_policy=OpenMoeForCausalLMPolicy(),
        )
    elif parallel == "ep_zero":
        plugin = MoeHybridParallelPlugin(
            tp_size=1,
            pp_size=1,
            zero_stage=2,
            extra_dp_size=2,
            custom_policy=OpenMoeForCausalLMPolicy(),
        )
    elif parallel == "hybrid":
        plugin = MoeHybridParallelPlugin(
            tp_size=1,
            pp_size=2,
            zero_stage=1,
            microbatch_size=1,
            custom_policy=OpenMoeForCausalLMPolicy(),
        )
    booster = Booster(plugin=plugin)
    model, optim, _, _, _ = booster.boost(model=model, optimizer=optim)
    return model, booster, optim


def _test_moe_checkpoint(rank, parallel, shard):
    if parallel == None:
        MOE_MANAGER.setup(
            parallel=None,
        )
    elif parallel == "ep":
        MOE_MANAGER.setup(
            parallel="EP",
        )
    elif parallel == "ep_zero":
        MOE_MANAGER.setup(
            parallel="EP",
            max_ep_size=2,
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
    if shard:
        booster1.save_model(model1, "./tmp_ckpt", shard=True, size_per_shard=1)
        booster2.load_model(model2, "./tmp_ckpt")
    else:
        booster1.save_model(model1, "./tmp_ckpt.pth")
        booster2.load_model(model2, "./tmp_ckpt.pth")
    check_state_dict_equal(model1.state_dict(), model2.state_dict(), False)
    if dist.get_rank() == 0:
        if shard:
            shutil.rmtree("./tmp_ckpt")
        else:
            os.remove("./tmp_ckpt.pth")

    # optim ckpt
    criterion = lambda x: x.mean()
    for _ in range(2):
        data = torch.randint(0, 4, (4, 16)).cuda()
        label = torch.randint(0, 4, (4,)).cuda()
        if parallel == "hybrid":
            kwargs = {"pipeline": True, "booster": booster1, "plugin": booster1.plugin}
        else:
            kwargs = {}
        run_fwd_bwd(model1, data, label, criterion, optim1, **kwargs)
        optim1.step()
        optim1.zero_grad()
    if shard:
        booster1.save_optimizer(optim1, "./tmp_ckpt", shard=True, size_per_shard=1)
        booster2.load_optimizer(optim2, "./tmp_ckpt")
    else:
        booster1.save_optimizer(optim1, "./tmp_ckpt.pth")
        booster2.load_optimizer(optim2, "./tmp_ckpt.pth")

    check_state_dict_equal(optim1.optim.state_dict(), optim2.optim.state_dict(), False)
    if dist.get_rank() == 0:
        if shard:
            shutil.rmtree("./tmp_ckpt")
        else:
            os.remove("./tmp_ckpt.pth")


def _run_dist(rank, world_size, port, parallel, shard):
    colossalai.launch(
        config=dict(),
        rank=rank,
        world_size=world_size,
        host="localhost",
        port=port,
        backend="nccl",
    )
    _test_moe_checkpoint(rank, parallel, shard)


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [4])
@pytest.mark.parametrize("parallel", [None, "ep", "ep_zero", "hybrid"])
@pytest.mark.parametrize("shard", [True, False])
@rerun_if_address_is_in_use()
def test_moe_checkpoint(world_size, parallel, shard):
    spawn(_run_dist, world_size, parallel=parallel, shard=shard)


if __name__ == "__main__":
    test_moe_checkpoint(world_size=4, parallel="hybrid", shard=False)
