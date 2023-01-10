import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

from colossalai.utils.cuda import get_current_device

from .registry import non_distributed_component_funcs
from .utils.dummy_data_generator import DummyDataGenerator


class DummyDataLoader(DummyDataGenerator):
    vocab_size = 128
    batch_size = 4
    seq_len = 64

    def generate(self):
        input_ids = torch.randint(0,
                                  DummyDataLoader.vocab_size, (DummyDataLoader.batch_size, DummyDataLoader.seq_len),
                                  device=get_current_device())
        return input_ids, input_ids


class GPTLMModel(nn.Module):

    def __init__(self,
                 hidden_size=768,
                 num_layers=12,
                 num_attention_heads=12,
                 max_seq_len=1024,
                 vocab_size=50304,
                 checkpoint=False):
        super().__init__()
        self.checkpoint = checkpoint
        self.model = GPT2LMHeadModel(
            GPT2Config(n_embd=hidden_size,
                       n_layer=num_layers,
                       n_head=num_attention_heads,
                       n_positions=max_seq_len,
                       n_ctx=max_seq_len,
                       vocab_size=vocab_size,
                       resid_pdrop=0.0,
                       embd_pdrop=0.0,
                       attn_pdrop=0.0))
        if checkpoint:
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids):
        # Only return lm_logits
        attention_mask = torch.ones_like(input_ids)
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=not self.checkpoint)[0]


def gpt2_micro(checkpoint=True):
    return GPTLMModel(checkpoint=checkpoint,
                      hidden_size=32,
                      num_layers=2,
                      num_attention_heads=4,
                      max_seq_len=64,
                      vocab_size=128)


def gpt2_s(checkpoint=True):
    return GPTLMModel(checkpoint=checkpoint)


def gpt2_m(checkpoint=True):
    return GPTLMModel(hidden_size=1024, num_layers=24, num_attention_heads=16, checkpoint=checkpoint)


class GPTLMLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


@non_distributed_component_funcs.register(name='gpt2')
def get_training_components():

    trainloader = DummyDataLoader()
    testloader = DummyDataLoader()

    criterion = GPTLMLoss()
    return gpt2_micro, trainloader, testloader, torch.optim.Adam, criterion
