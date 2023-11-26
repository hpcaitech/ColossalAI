from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel


## Define the Model and Loss Based on Huggingface transformers GPT2LMHeadModel
class GPTLMModel(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        num_layers=12,
        num_attention_heads=12,
        max_seq_len=1024,
        vocab_size=50257,
        checkpoint=False,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.config = GPT2Config(
            n_embd=hidden_size,
            n_layer=num_layers,
            n_head=num_attention_heads,
            n_positions=max_seq_len,
            n_ctx=max_seq_len,
            vocab_size=vocab_size,
        )
        self.model = GPT2LMHeadModel(self.config)
        if checkpoint:
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask):
        # Only return lm_logits
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=not self.checkpoint)[0]


def gpt2_medium(checkpoint=False):
    return GPTLMModel(hidden_size=1024, num_layers=24, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_xl(checkpoint=True):
    return GPTLMModel(hidden_size=1600, num_layers=48, num_attention_heads=32, checkpoint=checkpoint)


def gpt2_10b(checkpoint=True):
    return GPTLMModel(hidden_size=4096, num_layers=50, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_14b(checkpoint=True):
    return GPTLMModel(hidden_size=4096, num_layers=70, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_20b(checkpoint=True):
    return GPTLMModel(hidden_size=8192, num_layers=25, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_24b(checkpoint=True):
    return GPTLMModel(hidden_size=8192, num_layers=30, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_30b(checkpoint=True):
    return GPTLMModel(hidden_size=8192, num_layers=37, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_40b(checkpoint=True):
    return GPTLMModel(hidden_size=8192, num_layers=50, num_attention_heads=16, checkpoint=checkpoint)


def model_builder(model_size: str) -> callable:
    if model_size == "gpt2_medium":
        return gpt2_medium
    elif model_size == "gpt2_xl":
        return gpt2_xl
    elif model_size == "gpt2_10b":
        return gpt2_10b
    elif model_size == "gpt2_14b":
        return gpt2_14b
    elif model_size == "gpt2_20b":
        return gpt2_20b
    elif model_size == "gpt2_24b":
        return gpt2_24b
    elif model_size == "gpt2_30b":
        return gpt2_30b
    elif model_size == "gpt2_40b":
        return gpt2_40b
    else:
        raise TypeError(f"model_builder {model_size}")


__all__ = ["model_builder"]
