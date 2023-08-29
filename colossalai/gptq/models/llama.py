from dataclasses import dataclass, field, fields


@dataclass
class GPTQLlamaConfig():
    layer_name = "LlamaDecoderLayer"
    layer_blocks = "model.layers"
    linear_names = [["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"], ["self_attn.o_proj"],
                    ["mlp.up_proj", "mlp.gate_proj"], ["mlp.down_proj"]]
    model_names = ["model.embed_tokens", "model.norm"]
    attention = "self_attn"
    mlp = "mlp"


def reset_llama_attention_params(layer, tp_size=1):
    attention = getattr(layer, "self_attn")
    attention.hidden_size = attention.hidden_size // tp_size
    attention.num_heads = attention.num_heads // tp_size
    attention.num_key_value_heads = attention.num_key_value_heads // tp_size
