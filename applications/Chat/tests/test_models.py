import copy
from typing import Any, Callable, Dict, Tuple

import pytest
import torch
import torch.nn as nn
from coati.models.base import Actor, Critic, RewardModel, get_base_model
from coati.models.bloom import BLOOMRM, BLOOMActor, BLOOMCritic
from coati.models.generation import generate
from coati.models.gpt import GPTRM, GPTActor, GPTCritic
from coati.models.llama import LlamaActor, LlamaCritic, LlamaRM
from coati.models.lora import LoraLinear, convert_to_lora_module
from coati.models.loss import GPTLMLoss, LogExpLoss, LogSigLoss, PolicyLoss, ValueLoss
from coati.models.opt import OPTRM, OPTActor, OPTCritic
from coati.models.utils import calc_action_log_probs, compute_reward, masked_mean


@pytest.mark.gpu
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("seq_len", [32])
@pytest.mark.parametrize("actor_maker", [
    lambda: BLOOMActor(),
    lambda: GPTActor(),
    # HACK: skip llama due to long execution time
    # lambda: LlamaActor(),
    lambda: OPTActor()
])
@pytest.mark.parametrize("generate_kwargs", [{
    "max_length": 64,
    "use_cache": True,
    "do_sample": True,
    "temperature": 1.0,
    "top_k": 50,
}])
def test_generation(actor_maker: Callable[[], Actor],
                    batch_size: int,
                    seq_len: int,
                    generate_kwargs: Dict[str, Any]
                    ):
    actor = actor_maker()
    input_ids = torch.randint(0, 100, (batch_size, seq_len)).cuda()
    sequences = generate(actor.cuda(), input_ids, **generate_kwargs)
    assert sequences.shape == (batch_size, generate_kwargs["max_length"])


@pytest.mark.cpu
def test_utils():
    fn_input = {
        "tensor": torch.ones((10, )),
        "mask": torch.randint(0, 2, (10, ))
    }
    fn_output = masked_mean(dim=0, **fn_input)
    assert fn_output.dim() == 0
    assert torch.allclose(fn_output, torch.tensor(1.0))

    batch_size = 4
    num_labels = 10
    fn_input = {
        "r": torch.ones((batch_size, )),
        "kl_coef": 1.0,
        "log_probs": torch.randn((batch_size, num_labels)),
        "log_probs_base": torch.randn((batch_size, num_labels)),
        "action_mask": torch.randint(0, 2, (batch_size, num_labels))
    }
    fn_output = compute_reward(**fn_input)
    assert fn_output.shape == (batch_size, )

    batch_size = 4
    seq_len = 32
    num_labels = 10
    num_actions = 2
    fn_input = {
        "output": {
            "logits": torch.randn((batch_size, seq_len, num_labels))
        },
        "sequences": torch.randint(0, num_labels, (batch_size, seq_len)),
        "num_actions": num_actions,
    }
    fn_output = calc_action_log_probs(**fn_input)
    assert fn_output.shape == (batch_size, num_actions)


@pytest.mark.cpu
@pytest.mark.parametrize("lora_rank", [4])
@pytest.mark.parametrize("num_dim", [32])
@pytest.mark.parametrize("num_layers", [4])
def test_lora(lora_rank: int,
              num_dim: int,
              num_layers: int):
    model = nn.ModuleList(
        [nn.Linear(num_dim, num_dim)
         for _ in range(num_layers)]
    )
    lora_model = convert_to_lora_module(model, lora_rank)
    assert isinstance(lora_model, nn.ModuleList)
    for i in range(num_layers):
        assert isinstance(lora_model[i], LoraLinear)
        assert lora_model[i].lora_A.shape == (lora_rank, num_dim)
        assert lora_model[i].lora_B.shape == (num_dim, lora_rank)

    old_model = copy.deepcopy(lora_model)
    for i in range(num_layers):
        assert isinstance(lora_model[i], LoraLinear)
        assert torch.allclose(old_model[i].weight, lora_model[i].weight)
        assert torch.allclose(old_model[i].bias, lora_model[i].bias)
        assert torch.allclose(old_model[i].lora_B @ old_model[i].lora_A,
                              lora_model[i].lora_B @ lora_model[i].lora_A)
    optimizer = torch.optim.Adam(lora_model.parameters())
    x = torch.randn(8, num_dim)
    for i in range(num_layers):
        x = lora_model[i](x)
    loss = x.sum()
    loss.backward()
    optimizer.step()
    for i in range(num_layers):
        assert isinstance(lora_model[i], LoraLinear)
        assert torch.allclose(old_model[i].weight, lora_model[i].weight)
        assert torch.allclose(old_model[i].bias, lora_model[i].bias)
        assert not torch.allclose(old_model[i].lora_B @ old_model[i].lora_A,
                                  lora_model[i].lora_B @ lora_model[i].lora_A)


@pytest.mark.cpu
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("seq_len", [128])
@pytest.mark.parametrize("models_maker", [
    lambda: (BLOOMActor(), BLOOMCritic(), BLOOMRM()),
    lambda: (GPTActor(), GPTCritic(), GPTRM()),
    # HACK: skip llama due to long execution time
    # lambda: (LlamaActor(), LlamaCritic(), LlamaRM()),
    lambda: (OPTActor(), OPTCritic(), OPTRM()),
])
@torch.no_grad()
def test_models(models_maker: Callable[[], Tuple[Actor, Critic, RewardModel]],
                batch_size: int,
                seq_len: int):

    actor_input = {
        "input_ids": torch.randint(0, 100, (batch_size, seq_len)),
        "attention_mask": torch.randint(0, 2, (batch_size, seq_len))
    }
    critic_input = {
        "sequences": torch.randint(0, 100, (batch_size, seq_len)),
        "action_mask": torch.randint(0, 2, (batch_size, seq_len)),
        "attention_mask": torch.randint(0, 2, (batch_size, seq_len))
    }
    rm_input = {
        "sequences": torch.randint(0, 100, (batch_size, seq_len)),
        "attention_mask": torch.randint(0, 2, (batch_size, seq_len))
    }

    actor, critic, rm = models_maker()
    assert isinstance(actor, Actor)
    base_actor_model = get_base_model(actor)
    assert isinstance(critic, Critic)
    base_critic_model = get_base_model(critic)
    assert isinstance(rm, RewardModel)
    base_rm_model = get_base_model(rm)

    actor_output = actor(**actor_input)
    critic_output = critic(**critic_input)
    rm_output = rm(**rm_input)

    assert actor_output.logits.shape[:2] == (batch_size, seq_len)
    assert critic_output.shape == (batch_size, )
    assert rm_output.shape == (batch_size, )


@pytest.mark.cpu
@pytest.mark.parametrize("batch_size", [16])
@pytest.mark.parametrize("seq_len", [128])
@pytest.mark.parametrize("num_labels", [100])
def test_loss(batch_size: int,
              seq_len: int,
              num_labels: int):
    loss = GPTLMLoss()
    loss_input = {
        "logits": torch.randn(batch_size, seq_len, num_labels),
        "labels": torch.randint(0, num_labels, (batch_size, seq_len))
    }
    loss_output = loss(**loss_input)

    loss = PolicyLoss()
    loss_input = {
        "log_probs": torch.randn(batch_size, ),
        "old_log_probs": torch.randn(batch_size, ),
        "advantages": torch.randn(batch_size, )
    }
    loss_output = loss(**loss_input)

    loss = ValueLoss()
    loss_input = {
        "values": torch.randn(batch_size, ),
        "old_values": torch.randn(batch_size, ),
        "reward": torch.randn(batch_size, )
    }
    loss_output = loss(**loss_input)

    loss = LogSigLoss()
    loss_input = {
        "chosen_reward": torch.randn(batch_size, ),
        "reject_reward": torch.randn(batch_size, ),
    }
    loss_output = loss(**loss_input)

    loss = LogExpLoss()
    loss_input = {
        "chosen_reward": torch.randn(batch_size, ),
        "reject_reward": torch.randn(batch_size, ),
    }
    loss_output = loss(**loss_input)


if __name__ == "__main__":
    generate_kwargs = dict(max_length=40,
                           use_cache=True,
                           do_sample=True,
                           temperature=1.0,
                           top_k=50)
    test_generation(lambda: LlamaActor(),
                    batch_size=4,
                    seq_len=32,
                    generate_kwargs=generate_kwargs)

    test_utils()

    test_lora(lora_rank=2, num_dim=8, num_layers=2)

    test_models(models_maker=lambda: (BLOOMActor(),
                                      BLOOMCritic(),
                                      BLOOMRM()),
                batch_size=8,
                seq_len=128)

    test_loss(batch_size=8, seq_len=128, num_labels=100)
