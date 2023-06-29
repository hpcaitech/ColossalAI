import os
import tempfile
from contextlib import nullcontext

import pytest
import torch
import torch.distributed as dist
from coati.models.gpt import GPTActor
from coati.models.utils import calc_action_log_probs
from coati.trainer.strategies import DDPStrategy, GeminiStrategy, LowLevelZeroStrategy
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

from colossalai.nn.optimizer import HybridAdam
from colossalai.testing import rerun_if_address_is_in_use, spawn

GPT_CONFIG = GPT2Config(n_embd=128, n_layer=4, n_head=4)


def get_data(batch_size: int, seq_len: int = 10) -> dict:
    input_ids = torch.randint(0, 50257, (batch_size, seq_len), device='cuda')
    attention_mask = torch.ones_like(input_ids)
    return dict(input_ids=input_ids, attention_mask=attention_mask)


def run_test_checkpoint(strategy):
    BATCH_SIZE = 2

    if strategy == 'ddp':
        strategy = DDPStrategy()
    elif strategy == 'colossalai_gemini':
        strategy = GeminiStrategy(placement_policy='cuda', initial_scale=2**5)
    elif strategy == 'colossalai_zero2':
        strategy = LowLevelZeroStrategy(stage=2, placement_policy='cuda')
    else:
        raise ValueError(f'Unsupported strategy "{strategy}"')

    with strategy.model_init_context():
        actor = GPTActor(config=GPT_CONFIG).cuda()

    actor_optim = HybridAdam(actor.parameters())

    actor, actor_optim = strategy.prepare((actor, actor_optim))

    def run_step():
        data = get_data(BATCH_SIZE)
        action_mask = torch.ones_like(data['attention_mask'], dtype=torch.bool)
        actor_output = actor(data['input_ids'], data['attention_mask'])
        action_log_probs = calc_action_log_probs(actor_output, data['input_ids'], action_mask.size(1))
        loss = action_log_probs.sum()
        strategy.backward(loss, actor, actor_optim)
        strategy.optimizer_step(actor_optim)

    run_step()

    ctx = tempfile.TemporaryDirectory() if dist.get_rank() == 0 else nullcontext()

    with ctx as dirname:
        rank0_dirname = [dirname]
        dist.broadcast_object_list(rank0_dirname)
        rank0_dirname = rank0_dirname[0]

        model_path = os.path.join(rank0_dirname, 'model.pt')
        strategy.save_model(actor, model_path, only_rank0=True)

        optim_path = os.path.join(rank0_dirname, f'optim.pt')
        strategy.save_optimizer(actor_optim, optim_path, only_rank0=True)

        # FIXME(cwher): Sharded optimizer checkpoint is not supported yet.
        #  at "ColossalAI/colossalai/checkpoint_io/general_checkpoint_io.py", line 62
        # optim_path = os.path.join(rank0_dirname, f'optim-r{dist.get_rank()}.pt')
        # strategy.save_optimizer(actor_optim, optim_path, only_rank0=False)

        dist.barrier()

        strategy.load_model(actor, model_path, strict=False)
        strategy.load_optimizer(actor_optim, optim_path)

        dist.barrier()

    run_step()


def run_dist(rank, world_size, port, strategy):
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    run_test_checkpoint(strategy)


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [2])
@pytest.mark.parametrize('strategy', ['ddp', 'colossalai_zero2', 'colossalai_gemini'])
@rerun_if_address_is_in_use()
def test_checkpoint(world_size, strategy):
    spawn(run_dist, world_size, strategy=strategy)


if __name__ == '__main__':
    test_checkpoint(2, 'colossalai_zero2')
