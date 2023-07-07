from typing import Optional, Tuple, Union

import torch

__all__ = ['get_gpt2_forward']

def get_gpt2_forward():

    try:
        from xformers.ops import memory_efficient_attention as me_attention
        from xformers.ops.fmha.attn_bias import LowerTriangularMask
    except:
        raise ImportError("Error: xformers module is not installed. Please install it to use flash attention.")
        
    def gpt2_flash_attention_forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        _, tgt_len, _ = hidden_states.size()
        assert tgt_len % 4 == 0, "Flash Attention Error: The sequence length should be a multiple of 4."

        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = split_heads(query, self.num_heads, self.head_dim)
        key = split_heads(key, self.num_heads, self.head_dim)
        value = split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)

        if use_cache is True:
            present = (key, value)
        else:
            present = None
        
        attn_bias = None
        if not self.is_cross_attention:
            attn_bias = LowerTriangularMask()
        if attention_mask != None:
            if attn_bias:
                attn_bias.add_bias(attention_mask)
            else:
                batch_size, _, tgt_len, src_len = attention_mask.size()
                attn_bias = attention_mask.expand(batch_size, self.num_heads, tgt_len, src_len).contiguous()
        scale = value.size(-1) ** -0.5
        attn_output = me_attention(query=query, key=key, value=value, attn_bias=attn_bias, p=self.attn_dropout.p, scale=scale)
        attn_output = merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        outputs = (attn_output, present, None)

        return outputs
    
    return gpt2_flash_attention_forward

def split_heads(tensor, num_heads, attn_head_size):
    """
    Splits hidden_size dim into attn_head_size and num_heads
    """
    new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
    tensor = tensor.view(new_shape)
    return tensor

def merge_heads(tensor, num_heads, attn_head_size):
    """
    Merges attn_head_size dim and num_attn_heads dim into hidden_size
    """
    new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
    return tensor.view(new_shape)