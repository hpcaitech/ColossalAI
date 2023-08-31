from dataclasses import dataclass, field, fields


@dataclass
class GPTQBloomConfig():
    layer_name = "BloomBlock"
    layer_blocks = "transformer.h"
    linear_names = [["self_attention.query_key_value"], ["self_attention.dense"], ["mlp.dense_h_to_4h"],
                    ["mlp.dense_4h_to_h"]]
    model_names = ["transformer.word_embeddings", "transformer.word_embeddings_layernorm", "transformer.ln_f"]
    attention = "self_attention"
    mlp = "mlp"


def reset_bloom_attention_params(layer, tp_size=1):
    attention = getattr(layer, "self_attention")
    attention.hidden_size = attention.hidden_size // tp_size
    attention.num_heads = attention.num_heads // tp_size
