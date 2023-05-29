import torch.nn as nn
from transformers import OPTConfig, OPTForCausalLM

from tests.test_elixir.utils.registry import TEST_MODELS

from .gpt import micro_data_fn


class OPTLMModel(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.module = OPTForCausalLM(config=config)
        self.enable_gc = False

    def gradient_checkpointing_enable(self):
        self.module.gradient_checkpointing_enable()
        self.enable_gc = True

    def forward(self, input_ids, attention_mask):
        loss = self.module(
        # pre-commit: do not rearrange
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
            use_cache=(not self.enable_gc))['loss']
        return loss


def opt_micro():
    opt_config = OPTConfig(
    # pre-commit: do not rearrange
        vocab_size=128,
        activation_dropout=0.0,
        dropout=0,
        hidden_size=32,
        num_hidden_layers=4,
        ffn_dim=128,
        num_attention_heads=4,
        word_embed_proj_dim=32,
        output_projection=True)
    return OPTLMModel(opt_config)


TEST_MODELS.register('opt_micro', opt_micro, micro_data_fn)
