import os
from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
from coati.experience_maker import NaiveExperienceMaker
from coati.models.base import RewardModel
from coati.models.gpt import GPTActor, GPTCritic
from coati.replay_buffer import NaiveReplayBuffer
from coati.trainer.strategies import DDPStrategy, GeminiStrategy
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

from colossalai.testing import rerun_if_address_is_in_use, spawn

GPT_CONFIG = GPT2Config(n_embd=128, n_layer=4, n_head=4)


def get_data(batch_size: int, seq_len: int = 10) -> dict:
    input_ids = torch.randint(0, 50257, (batch_size, seq_len), device='cuda')
    attention_mask = torch.ones_like(input_ids)
    return dict(input_ids=input_ids, attention_mask=attention_mask)


def gather_and_equal(tensor: torch.Tensor) -> bool:
    world_size = dist.get_world_size()
    outputs = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(outputs, tensor.contiguous())
    for t in outputs[1:]:
        if not torch.equal(outputs[0], t):
            return False
    return True


def run_test_data(strategy):
    EXPERIENCE_BATCH_SIZE = 4
    SAMPLE_BATCH_SIZE = 2

    if strategy == 'ddp':
        strategy = DDPStrategy()
    elif strategy == 'colossalai':
        strategy = GeminiStrategy(placement_policy='cuda')
    else:
        raise ValueError(f'Unsupported strategy "{strategy}"')

    actor = GPTActor(config=GPT_CONFIG).cuda()
    critic = GPTCritic(config=GPT_CONFIG).cuda()

    initial_model = deepcopy(actor)
    reward_model = RewardModel(deepcopy(critic.model)).cuda()

    experience_maker = NaiveExperienceMaker(actor, critic, reward_model, initial_model)
    replay_buffer = NaiveReplayBuffer(SAMPLE_BATCH_SIZE, cpu_offload=False)

    # experience of all ranks should be the same
    for _ in range(2):
        data = get_data(EXPERIENCE_BATCH_SIZE)
        assert gather_and_equal(data['input_ids'])
        assert gather_and_equal(data['attention_mask'])
        experience = experience_maker.make_experience(**data,
                                                      do_sample=True,
                                                      max_length=16,
                                                      eos_token_id=50256,
                                                      pad_token_id=50256)
        assert gather_and_equal(experience.sequences)
        assert gather_and_equal(experience.action_log_probs)
        assert gather_and_equal(experience.values)
        assert gather_and_equal(experience.reward)
        assert gather_and_equal(experience.advantages)
        assert gather_and_equal(experience.action_mask)
        assert gather_and_equal(experience.attention_mask)
        replay_buffer.append(experience)

    # replay buffer's data should be the same
    buffer_size = torch.tensor([len(replay_buffer)], device='cuda')
    assert gather_and_equal(buffer_size)
    for item in replay_buffer.items:
        assert gather_and_equal(item.sequences)
        assert gather_and_equal(item.action_log_probs)
        assert gather_and_equal(item.values)
        assert gather_and_equal(item.reward)
        assert gather_and_equal(item.advantages)
        assert gather_and_equal(item.action_mask)
        assert gather_and_equal(item.attention_mask)

    # dataloader of each rank should have the same size and different batch
    dataloader = strategy.setup_dataloader(replay_buffer)
    dataloader_size = torch.tensor([len(dataloader)], device='cuda')
    assert gather_and_equal(dataloader_size)
    for experience in dataloader:
        assert not gather_and_equal(experience.sequences)
        assert not gather_and_equal(experience.action_log_probs)
        assert not gather_and_equal(experience.values)
        assert not gather_and_equal(experience.reward)
        assert not gather_and_equal(experience.advantages)
        # action mask and attention mask may be same


def run_dist(rank, world_size, port, strategy):
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    run_test_data(strategy)


@pytest.mark.skip
@pytest.mark.dist
@pytest.mark.parametrize('world_size', [2])
@pytest.mark.parametrize('strategy', ['ddp', 'colossalai'])
@rerun_if_address_is_in_use()
def test_data(world_size, strategy):
    spawn(run_dist, world_size, strategy=strategy)


if __name__ == '__main__':
    test_data(2, 'colossalai')
