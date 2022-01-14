from transformers import GPT2LMHeadModel, GPT2Config
import torch.nn as nn

__all__ = [
    'GPTLMModel',
    'GPTLMLoss',
    'GPT2_small',
    'GPT2_medium',
    'GPT2_large',
    'GPT2_exlarge',
    'GPT3'
]

# vanilla GPT model adapted from huggingface transformers


class GPTLMModel(nn.Module):
    def __init__(self, hidden_size=768, num_layers=12, num_attention_heads=12, max_seq_len=1024, checkpoint=False):
        super().__init__()
        self.checkpoint = checkpoint
        self.model = GPT2LMHeadModel(GPT2Config(n_embd=hidden_size, n_layer=num_layers,
                                     n_head=num_attention_heads, n_positions=max_seq_len, n_ctx=max_seq_len))
        if checkpoint:
            self.model.transformer.gradient_checkpointing = True

    def forward(self, input_ids, attention_mask):
        # Only return lm_logits
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=not self.checkpoint)[0]


class GPTLMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def GPT2_small(checkpoint=False):
    cfg = dict(hidden_size=768, num_layers=12, num_attention_heads=12)
    return GPTLMModel(checkpoint=checkpoint, **cfg)


def GPT2_medium(checkpoint=False):
    cfg = dict(hidden_size=1024, num_layers=24, num_attention_heads=16)
    return GPTLMModel(checkpoint=checkpoint, **cfg)


def GPT2_large(checkpoint=False):
    cfg = dict(hidden_size=1280, num_layers=36, num_attention_heads=20)
    return GPTLMModel(checkpoint=checkpoint, **cfg)


def GPT2_exlarge(checkpoint=False):
    cfg = dict(hidden_size=1600, num_layers=48, num_attention_heads=25)
    return GPTLMModel(checkpoint=checkpoint, **cfg)


def GPT3(checkpoint=False):
    cfg = dict(hidden_size=12288, num_layers=96, num_attention_heads=96, max_seq_len=2048)
    return GPTLMModel(checkpoint=checkpoint, **cfg)
