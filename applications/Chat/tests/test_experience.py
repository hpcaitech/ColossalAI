import copy
import os

import pytest
import torch
import torch.distributed as dist
from coati.experience_buffer import NaiveExperienceBuffer
from coati.experience_maker import NaiveExperienceMaker
from coati.models.base import RewardModel
from coati.models.gpt import GPTActor, GPTCritic
from coati.trainer.ppo import _set_default_generate_kwargs
from coati.trainer.strategies import DDPStrategy, GeminiStrategy
from coati.trainer.strategies.colossalai import LowLevelZeroStrategy
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

from colossalai.testing import rerun_if_address_is_in_use, spawn

GPT_CONFIG = GPT2Config(n_embd=128, n_layer=4, n_head=4)


def get_data(batch_size: int, seq_len: int = 10) -> dict:
    input_ids = torch.randint(0, 50257, (batch_size, seq_len), device="cuda")
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


def make_and_consume_experience(strategy):
    EXPERIENCE_BATCH_SIZE = 4
    SAMPLE_BATCH_SIZE = 2

    if strategy == "ddp":
        strategy = DDPStrategy()
    elif strategy == "colossalai-zero2":
        strategy = LowLevelZeroStrategy()
    elif strategy == "colossalai-gemini":
        strategy = GeminiStrategy(placement_policy="static")
    else:
        raise ValueError(f'Unsupported strategy "{strategy}"')

    with strategy.model_init_context():
        actor = GPTActor(config=GPT_CONFIG).cuda()
        critic = GPTCritic(config=GPT_CONFIG).cuda()

        initial_model = GPTActor(config=GPT_CONFIG).cuda()
        reward_model = RewardModel(model=copy.deepcopy(critic.model)).cuda()

    actor, critic, initial_model, reward_model = strategy.prepare(actor, critic, initial_model, reward_model)

    class MockTokenizer:
        def __init__(self):
            self.padding_side = "left"
            self.eos_token_id = 0
            self.pad_token_id = 0

        def batch_decode(self, sequences, skip_special_tokens=True):
            return ["This is a test sentence." for i in range(len(sequences))]

        def __call__(self, sequences, **kwargs):
            return {
                "input_ids": torch.randint(0, 50257, (len(sequences), 100), device="cpu"),
                "attention_mask": torch.ones((len(sequences), 100), device="cpu").bool(),
            }

    tokenizer = MockTokenizer()
    experience_maker = NaiveExperienceMaker(actor, critic, reward_model, initial_model, tokenizer, tokenizer)
    data_buffer = NaiveExperienceBuffer(SAMPLE_BATCH_SIZE, cpu_offload=False)

    generate_kwargs = dict(do_sample=True, max_length=16)
    generate_kwargs = _set_default_generate_kwargs(strategy, generate_kwargs, actor)

    # experience of all ranks should be the same
    for _ in range(2):
        data = get_data(EXPERIENCE_BATCH_SIZE)
        assert gather_and_equal(data["input_ids"])
        assert gather_and_equal(data["attention_mask"])
        experience = experience_maker.make_experience(**data, do_sample=True, max_length=16)
        assert gather_and_equal(experience.sequences)
        assert gather_and_equal(experience.action_log_probs)
        assert gather_and_equal(experience.values)
        assert gather_and_equal(experience.reward)
        assert gather_and_equal(experience.advantages)
        assert gather_and_equal(experience.action_mask)
        assert gather_and_equal(experience.attention_mask)
        data_buffer.append(experience)

    # data buffer's data should be the same
    buffer_size = torch.tensor([len(data_buffer)], device="cuda")
    assert gather_and_equal(buffer_size)
    for item in data_buffer.items:
        assert gather_and_equal(item.sequences)
        assert gather_and_equal(item.action_log_probs)
        assert gather_and_equal(item.values)
        assert gather_and_equal(item.reward)
        assert gather_and_equal(item.advantages)
        assert gather_and_equal(item.action_mask)
        assert gather_and_equal(item.attention_mask)

    # dataloader of each rank should have the same size and different batch
    dataloader = strategy.setup_dataloader(data_buffer)
    dataloader_size = torch.tensor([len(dataloader)], device="cuda")
    assert gather_and_equal(dataloader_size)
    for experience in dataloader:
        assert not gather_and_equal(experience.sequences)
        assert not gather_and_equal(experience.action_log_probs)
        assert not gather_and_equal(experience.values)
        assert not gather_and_equal(experience.reward)
        assert not gather_and_equal(experience.advantages)
        # action mask and attention mask may be same


def run_dist(rank, world_size, port, strategy):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    make_and_consume_experience(strategy)


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [2])
@pytest.mark.parametrize("strategy", ["ddp", "colossalai-zero2", "colossalai-gemini"])
@rerun_if_address_is_in_use()
def test_experience(world_size, strategy):
    spawn(run_dist, world_size, strategy=strategy)


if __name__ == "__main__":
    test_experience(2, "colossalai-zero2")
