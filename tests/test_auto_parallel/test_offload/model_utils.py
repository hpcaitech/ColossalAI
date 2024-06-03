import torch
import torch.nn as nn
from transformers import BertConfig, BertLMHeadModel, GPT2Config, GPT2LMHeadModel

# from tests.components_to_test.registry import non_distributed_component_funcs


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


class LMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


class BertLMModel(nn.Module):
    def __init__(self, hidden_size=768, num_layers=12, num_attention_heads=32, vocab_size=30522):
        super().__init__()
        self.model = BertLMHeadModel(
            BertConfig(
                n_embd=hidden_size,
                num_hidden_layers=num_layers,
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                max_position_embeddings=hidden_size,
                vocab_size=vocab_size,
            )
        )

    def forward(self, input_ids, attention_mask):
        # Only return lm_logits
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)[0]


# @non_distributed_component_funcs.register(name="bert_")
def get_bert_components():
    vocab_size = 1024
    seq_len = 64
    batchSize = 64

    def bert_model_builder():
        model = BertLMModel(hidden_size=8192, num_layers=4, num_attention_heads=32, vocab_size=vocab_size)
        return model

    def bert_data_gen(device="meta"):
        input_ids = torch.randint(0, vocab_size, (batchSize, seq_len), device=device)
        attention_mask = torch.ones_like(input_ids, device=device)
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        return kwargs

    return bert_model_builder, bert_data_gen


# @non_distributed_component_funcs.register(name="gpt2_")
def get_gpt2_components():
    vocab_size = 1024
    seq_len = 8
    batchSize = 64

    def gpt2_model_builder():
        model = GPTLMModel(hidden_size=8192, num_layers=2, num_attention_heads=32, vocab_size=vocab_size)
        return model

    def gpt2_data_gen(device="meta"):
        input_ids = torch.randint(0, vocab_size, (batchSize, seq_len), device=device)
        attention_mask = torch.ones_like(input_ids, device=device)
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        return kwargs

    return gpt2_model_builder, gpt2_data_gen
