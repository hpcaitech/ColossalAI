#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
from typing import Tuple
from colossalai.utils.activation_checkpoint import checkpoint
import torch
from torch import nn as nn, Tensor, distributed as dist
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import torch.nn.init as init
from colossalai.registry import MODELS, LOSSES
from colossalai.nn.layer.parallel_1d._utils import reduce_grad, reduce_input
from colossalai.context import seed, ParallelMode
from colossalai.core import global_context as gpc
from colossalai.nn.layer.utils import divide, ACT2FN
from colossalai.registry import LAYERS, LOSSES, MODELS
from colossalai.utils import checkpoint
from colossalai.utils import get_current_device
from colossalai.nn.layer import Linear1D_Col, Linear1D_Row
from colossalai.nn.layer.base_layer import ParallelLayer
from colossalai.kernel.jit import bias_gelu_impl


__all__ = [
    'BERTLMLoss',
]


@LOSSES.register_module
class BERTLMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn1 = nn.CrossEntropyLoss(reduction='none')
        self.loss_fn2 = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, sequence_output, nsp_prediction, masked_lm_positions, masked_lm_ids, masked_lm_weights, next_sentence_labels):
        # MLM loss
        vocab_size = sequence_output.shape[-1]
        masked_lm_positions = masked_lm_positions.unsqueeze(-1)
        masked_lm_positions = masked_lm_positions.expand(-1, -1, vocab_size)
        sequence_output = torch.gather(sequence_output, 1, masked_lm_positions)
        sequence_output = sequence_output.view(-1, vocab_size)
        masked_lm_ids = masked_lm_ids.view(-1)
        MLM_loss = self.loss_fn1(sequence_output, masked_lm_ids)
        MLM_loss = MLM_loss * masked_lm_weights.view(-1)
        MLM_loss = MLM_loss.mean()

        # NSP loss
        NSP_loss = self.loss_fn2(nsp_prediction.view(-1, 2).float(),
                                 next_sentence_labels.view(-1),)

        NSP_loss = NSP_loss.float()
        loss = MLM_loss + NSP_loss
        return loss


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, hidden_size, act_func: str = 'gelu', layer_norm_epsilon: float = 1e-6):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = ACT2FN[act_func]
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMHead(nn.Module):
    def __init__(self, hidden_size: int = 768, act_func: str = 'gelu', layer_norm_epsilon: float = 1e-6, vocab_size: int = 30522):
        super().__init__()
        self.transform = BertPredictionHeadTransform(hidden_size, act_func, layer_norm_epsilon)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertNSPHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.seq_relationship = nn.Linear(hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPretrainingHead(nn.Module):
    def __init__(self, hidden_size: int = 768, act_func: str = 'gelu', layer_norm_epsilon: float = 1e-6, vocab_size: int = 30522):
        super().__init__()
        self.predictions = BertLMHead(hidden_size, act_func, layer_norm_epsilon, vocab_size)
        self.seq_relationship = nn.Linear(hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


@MODELS.register_module
class BertModel1D(nn.Module):
    def __init__(self,
                 vocab_size: int = 30522,
                 depth: int = 12,
                 num_heads: int = 12,
                 hidden_size: int = 768,
                 mlp_ratio: int = 4.0,
                 max_position_embeddings: int = 512,
                 num_tokentypes: int = 2,
                 drop_rate: float = 0.1,
                 embed_drop_rate: float = 0.1,
                 attn_drop_rate: float = 0.1,
                 act_func: str = 'gelu',
                 checkpoint: bool = False,
                 dtype: torch.dtype = None,
                 padding_idx: int = 0,
                 layer_norm_epsilon: float = 1e-6,
                 apply_post_layer_norm: bool = False,
                 position_embedding_type: str = "absolute",
                 add_pooling_layer: bool = True):
        super().__init__()

        self.embed = BertEmbeddings(
            hidden_size,
            vocab_size,
            max_position_embeddings,
            num_tokentypes,
            embed_drop_rate,
            padding_idx,
            layer_norm_epsilon,
            position_embedding_type)

        self.blocks = nn.ModuleList([
            BertTransformerLayer1D(hidden_size, num_heads, act_func, mlp_ratio, attn_drop_rate,
                                   drop_rate, dtype, checkpoint, max_position_embeddings, layer_norm_epsilon, apply_post_layer_norm)
            for i in range(depth)
        ])
        self.pooler = BertPooler(hidden_size) if add_pooling_layer else None
        self.dtype = dtype

    def get_extended_attention_mask(self, attention_mask: Tensor) -> Tensor:
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # Adapted from huggingface
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length))

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long)

        hidden_states = self.embed(input_ids=input_ids, token_type_ids=token_type_ids)
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask)

        for block in self.blocks:
            hidden_states, extended_attention_mask = block(hidden_states, extended_attention_mask)

        pooled_output = self.pooler(hidden_states) if self.pooler is not None else None

        return (hidden_states, pooled_output)
        return hidden_states


class BertForPreTraining1D(nn.Module):
    def __init__(self,
                 vocab_size: int = 30522,
                 depth: int = 12,
                 num_heads: int = 12,
                 hidden_size: int = 768,
                 mlp_ratio: int = 4.0,
                 max_position_embeddings: int = 512,
                 num_tokentypes: int = 2,
                 drop_rate: float = 0.1,
                 embed_drop_rate: float = 0.1,
                 attn_drop_rate: float = 0.1,
                 act_func: str = 'gelu',
                 checkpoint: bool = False,
                 dtype: torch.dtype = None,
                 padding_idx: int = 0,
                 layer_norm_epsilon: float = 1e-6,
                 apply_post_layer_norm: bool = False,
                 position_embedding_type: str = "absolute"):
        super().__init__()

        self.bert = BertModel1D(
            vocab_size,
            depth,
            num_heads,
            hidden_size,
            mlp_ratio,
            max_position_embeddings,
            num_tokentypes,
            drop_rate,
            embed_drop_rate,
            attn_drop_rate,
            act_func,
            checkpoint,
            dtype,
            padding_idx,
            layer_norm_epsilon,
            apply_post_layer_norm,
            position_embedding_type)
        self.cls = BertPretrainingHead(hidden_size, act_func, layer_norm_epsilon, vocab_size)

        # Initialize weights and apply final processing
        # self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        next_sentence_label=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        # output = (prediction_scores, seq_relationship_score) + outputs[2:]
        output = (prediction_scores, seq_relationship_score)
        return output


@LAYERS.register_module
class BertMLP1D(ParallelLayer):
    """MLP layer for 1D parallel BERT

    :param in_features: size of each input sample
    :type in_features: int
    :param mlp_ratio: hidden size of MLP divided by embedding dim
    :type mlp_ratio: int
    :param act_func: activation function, defaults to 'gelu'
    :type act_func: str, optional
    :param dropout_prob: dropout probability, defaults to 0.
    :type dropout_prob: float, optional
    :param dtype: The dtype of parameters, defaults to None
    :type dtype: torch.dtype, optional
    :param checkpoint: whether to checkpoint the layer, defaults to False
    :type checkpoint: bool, optional
    """

    def __init__(self,
                 in_features: int,
                 mlp_ratio: int,
                 act_func: str = 'gelu',
                 dropout_prob: float = 0.,
                 dtype=None,
                 checkpoint: bool = False,
                 skip_bias_add: bool = False,
                 weight_init='torch'
                 ):
        super().__init__()

        self.in_features = in_features
        self.mlp_ratio = mlp_ratio
        self.checkpoint = checkpoint
        self.skip_bias_add = skip_bias_add
        assert weight_init in ('torch', 'jax')

        if act_func == 'fused_gelu':
            self.act = bias_gelu_impl
            skip_dense_1_add_bias = True
        else:
            self.act = ACT2FN[act_func]
            skip_dense_1_add_bias = False

        # Project to mlp_ratio * h.
        self.dense_1 = Linear1D_Col(
            self.in_features,
            int(self.mlp_ratio * self.in_features),
            dtype=dtype,
            gather_output=False,
            skip_bias_add=skip_dense_1_add_bias,
            init_weight=weight_init,
            init_bias=weight_init
        )

        # Project back to h.
        self.dense_2 = Linear1D_Row(
            int(self.mlp_ratio * self.in_features),
            self.in_features,
            dtype=dtype,
            parallel_input=True,
            init_weight=weight_init, init_bias=weight_init
        )

        self.dropout = nn.Dropout(dropout_prob)

    def _forward(self, hidden_states: Tensor) -> Tensor:
        if self.act == bias_gelu_impl:
            intermediate_output, bias = self.dense_1(hidden_states)
            intermediate_output = self.act(intermediate_output, bias)
        else:
            intermediate_output = self.dense_1(hidden_states)
            intermediate_output = self.act(intermediate_output)

        output = self.dense_2(intermediate_output)
        output = self.dropout(output)
        return output

    def _checkpoint_forward(self, hidden_states: Tensor) -> Tensor:
        return checkpoint(self._forward, hidden_states)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if self.checkpoint:
            return self._checkpoint_forward(hidden_states)
        else:
            return self._forward(hidden_states)


@LAYERS.register_module
class BertSelfAttention1D(ParallelLayer):
    """Self-attention layer for 1D parallel Bert
    """

    def __init__(self,
                 hidden_size: int,
                 num_attention_heads: int,
                 attention_dropout_prob: float,
                 hidden_dropout_prob: float,
                 dtype=None,
                 checkpoint: bool = False,
                 weight_init='torch',
                 max_position_embeddings=1024,
                 ):
        super().__init__()

        self.hidden_size = hidden_size
        self.attention_head_size = divide(hidden_size, num_attention_heads)
        self.num_attention_heads_per_partition = divide(num_attention_heads, gpc.tensor_parallel_size)
        self.hidden_size_per_partition = divide(hidden_size, gpc.tensor_parallel_size)

        self.checkpoint = checkpoint
        assert weight_init in ('torch', 'jax')
        if weight_init == 'jax':
            init_bias = 'zero'
        else:
            init_bias = weight_init

        self.query_key_value = Linear1D_Col(
            hidden_size,
            3 * hidden_size,
            dtype=dtype,
            init_weight=weight_init,
            init_bias=init_bias
        )
        self.attention_dropout = nn.Dropout(attention_dropout_prob)
        self.dense = Linear1D_Row(
            hidden_size,
            hidden_size,
            dtype=dtype,
            parallel_input=True,
            init_weight=weight_init, init_bias=init_bias
        )
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.softmax = nn.Softmax(dim=-1)

        max_positions = max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

    def _forward(self, hidden_states: Tensor, attention_mask=None) -> Tensor:
        query_key_value = self.query_key_value(hidden_states)
        new_qkv_shape = query_key_value.shape[:-1] + \
            (self.num_attention_heads_per_partition, 3 * self.attention_head_size)
        query_key_value = query_key_value.view(new_qkv_shape)
        query_key_value = query_key_value.permute((0, 2, 1, 3))
        query_layer, key_layer, value_layer = torch.chunk(
            query_key_value, 3, dim=-1)

        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask
            attention_scores = attention_scores + attention_mask

        attention_scores = self.softmax(attention_scores)
        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attention_scores = attention_scores.type(value_layer.dtype)

        with seed(ParallelMode.TENSOR):
            attention_probs = self.attention_dropout(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(1, 2)
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.reshape(new_context_layer_shape)
        output = self.dense(context_layer)
        output = self.dropout(output)

        return output

    def _checkpoint_forward(self, hidden_states: Tensor, attention_mask=None) -> Tensor:
        return checkpoint(self._forward, hidden_states, attention_mask)

    def forward(self, hidden_states: Tensor, attention_mask=None) -> Tensor:
        if self.checkpoint:
            return self._checkpoint_forward(hidden_states, attention_mask)
        else:
            return self._forward(hidden_states, attention_mask)


@LAYERS.register_module
class BertTransformerLayer1D(ParallelLayer):
    """Pre-Layernorm Transformer layer which contains a self-attention layer and a MLP layer.

    :param hidden_size: hidden size
    :type hidden_size: int
    :param num_attention_heads: number of attention heads
    :type num_attention_heads: int
    :param act_func: activation function, defaults to 'gelu'
    :type act_func: str, optional
    :param mlp_ratio: hidden size of MLP divided by embedding dim, defaults to 4.0
    :type mlp_ratio: float, optional
    :param attention_dropout_prob: dropout probability for attention layer, defaults to 0.
    :type attention_dropout_prob: float, optional
    :param hidden_dropout_prob: dropout probability for attention layer, defaults to 0.
    :type hidden_dropout_prob: float, optional
    :param dtype: dtype of parameters, defaults to None
    :type dtype: torch.dtype, optional
    """

    def __init__(self,
                 hidden_size: int,
                 num_attention_heads: int,
                 act_func: str = 'gelu',
                 mlp_ratio: float = 4.0,
                 attention_dropout_prob: float = 0.,
                 hidden_dropout_prob: float = 0.,
                 dtype=None,
                 checkpoint: bool = False,
                 max_position_embeddings: int = 1024,
                 layer_norm_epsilon: float = 1e-5,
                 apply_post_layer_norm: bool = False
                 ):
        super().__init__()

        self.dtype = dtype
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.apply_post_layer_norm = apply_post_layer_norm
        self.attention = BertSelfAttention1D(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout_prob=attention_dropout_prob,
            hidden_dropout_prob=hidden_dropout_prob,
            dtype=dtype,
            max_position_embeddings=max_position_embeddings,
            checkpoint=checkpoint,
        )

        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.mlp = BertMLP1D(
            in_features=hidden_size,
            dropout_prob=hidden_dropout_prob,
            act_func=act_func,
            mlp_ratio=mlp_ratio,
            dtype=dtype,
            checkpoint=checkpoint,
        )

    def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        if not self.apply_post_layer_norm:
            residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        if self.apply_post_layer_norm:
            residual = hidden_states
        attention_output = self.attention(hidden_states, attention_mask)
        hidden_states = residual + attention_output

        if not self.apply_post_layer_norm:
            residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        if self.apply_post_layer_norm:
            residual = hidden_states
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        output = (hidden_states, attention_mask)
        return output


# class BERTModelForPretraining(nn.Module):

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self,
                 hidden_size: int,
                 vocab_size: int,
                 max_position_embeddings: int,
                 num_tokentypes: int = 2,
                 embed_drop_prob: float = 0.1,
                 padding_idx: int = 0,
                 layer_norm_epsilon: float = 1e-6,
                 position_embedding_type: str = "absolute"
                 ):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(num_tokentypes, hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.dropout = nn.Dropout(embed_drop_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = position_embedding_type
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))
        # if version.parse(torch.__version__) > version.parse("1.6.0"):
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False,
        )

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
