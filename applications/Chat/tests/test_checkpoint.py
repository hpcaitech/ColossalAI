import os
import tempfile
from contextlib import nullcontext

import pytest
import torch
import torch.distributed as dist
from coati.models.gpt import GPTActor
from coati.models.utils import calc_action_log_probs
from coati.trainer.strategies import DDPStrategy, GeminiStrategy, LowLevelZeroStrategy, Strategy
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

from colossalai.nn.optimizer import HybridAdam
from colossalai.testing import rerun_if_address_is_in_use, spawn

GPT_CONFIG = GPT2Config(n_embd=128, n_layer=4, n_head=4)


def get_data(batch_size: int, seq_len: int = 10) -> dict:
    input_ids = torch.randint(0, 50257, (batch_size, seq_len), device="cuda")
    attention_mask = torch.ones_like(input_ids)
    return dict(input_ids=input_ids, attention_mask=attention_mask)


def train_step(strategy: Strategy, actor: GPTActor, actor_optim: HybridAdam, batch_size: int = 8):
    data = get_data(batch_size)
    action_mask = torch.ones_like(data["attention_mask"], dtype=torch.bool)
    actor_logits = actor(data["input_ids"], data["attention_mask"])["logits"]
    action_log_probs = calc_action_log_probs(actor_logits, data["input_ids"], action_mask.size(1))
    loss = action_log_probs.sum()
    strategy.backward(loss, actor, actor_optim)
    strategy.optimizer_step(actor_optim)


def run_test_checkpoint(strategy_name: str, shard: bool):
    if strategy_name == "ddp":
        strategy = DDPStrategy()
    elif strategy_name == "colossalai_gemini":
        strategy = GeminiStrategy(placement_policy="auto", initial_scale=2**5)
    elif strategy_name == "colossalai_zero2":
        strategy = LowLevelZeroStrategy(stage=2, placement_policy="cuda")
    else:
        raise ValueError(f"Unsupported strategy '{strategy_name}'")

    with strategy.model_init_context():
        actor = GPTActor(config=GPT_CONFIG).cuda()
    actor_optim = HybridAdam(actor.parameters())
    actor, actor_optim = strategy.prepare((actor, actor_optim))

    train_step(strategy, actor, actor_optim)

    ctx = tempfile.TemporaryDirectory() if dist.get_rank() == 0 else nullcontext()

    with ctx as dirname:
        rank0_dirname = [dirname]
        dist.broadcast_object_list(rank0_dirname)
        rank0_dirname = rank0_dirname[0]

        model_path = os.path.join(rank0_dirname, "model" if shard else f"model.pt")
        strategy.save_model(actor, model_path)
        optim_path = os.path.join(rank0_dirname, "optim" if shard else "optim.pt")
        strategy.save_optimizer(actor_optim, optim_path)
        dist.barrier()

        strategy.load_model(actor, model_path, strict=False)
        strategy.load_optimizer(actor_optim, optim_path)
        dist.barrier()

    train_step(strategy, actor, actor_optim)


def run_dist(rank: int, world_size: int, port: int, strategy_name: str, shard: bool):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    run_test_checkpoint(strategy_name, shard)


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [4])
@pytest.mark.parametrize("strategy_name", ["ddp", "colossalai_gemini", "colossalai_zero2"])
@pytest.mark.parametrize("shard", [False, True])
@rerun_if_address_is_in_use()
def test_checkpoint(world_size: int, strategy_name: str, shard: bool):
    spawn(run_dist, world_size, strategy_name=strategy_name, shard=shard)


if __name__ == "__main__":
    test_checkpoint(2, "colossalai_gemini", shard=False)
