import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel


class GPTLMModel(nn.Module):
    def __init__(self, hidden_size=768, num_layers=12, num_attention_heads=12, max_seq_len=1024, vocab_size=50257):
        super().__init__()
        self.model = GPT2LMHeadModel(
            GPT2Config(
                n_embd=hidden_size,
                n_layer=num_layers,
                n_head=num_attention_heads,
                n_positions=max_seq_len,
                n_ctx=max_seq_len,
                vocab_size=vocab_size,
            )
        )

    def forward(self, input_ids, attention_mask):
        # Only return lm_logits
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)[0]


class GPTLMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def get_gpt2_components(model_type: str, batch_size: int):
    vocab_size = 1024
    seq_len = 8

    def gpt2_model_builder():
        if model_type == "gpt2_medium":
            return GPTLMModel(hidden_size=1024, num_layers=24, num_attention_heads=16)
        elif model_type == "gpt2_xl":
            return GPTLMModel(hidden_size=1600, num_layers=48, num_attention_heads=32)
        elif model_type == "gpt2_10b":
            return GPTLMModel(hidden_size=4096, num_layers=50, num_attention_heads=16)
        elif model_type == "gpt2_14b":
            return GPTLMModel(hidden_size=4096, num_layers=70, num_attention_heads=16)
        elif model_type == "gpt2_20b":
            return GPTLMModel(hidden_size=8192, num_layers=25, num_attention_heads=16)
        elif model_type == "gpt2_24b":
            return GPTLMModel(hidden_size=8192, num_layers=30, num_attention_heads=16)
        else:
            raise TypeError(f"model_builder {model_type}")

    def gpt2_data_gen(device="cuda"):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        attention_mask = torch.ones_like(input_ids, device=device)
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        return kwargs

    return gpt2_model_builder, gpt2_data_gen
