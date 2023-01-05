import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.autograd import Function


def check_config(config):
    if config.hidden_size % config.nhead != 0:
        raise Exception("hidden_size % nhead != 0")

    factor = 8 if config.fp16 else 4
    upbound = factor * 1024 * 4
    if config.hidden_size > upbound:
        # as required by ln backward kernel currently
        raise Exception(f"hidden_size > {upbound}")

    head_dim = config.hidden_size // config.nhead
    if head_dim % factor != 0:
        # as required by reshape kernel
        raise Exception(f"head_dim({head_dim}) % {factor} != 0")


def calc_offset(sizes):
    offsets = [0]
    tmp = 0
    for x in sizes:
        tmp += x
        offsets.append(tmp)
    return offsets


colossal_multihead_attention = None


@dataclass
class Config:
    max_batch_tokens: int    # max batch token numbers
    max_seq_len: int    # max sequence length
    hidden_size: int    # size of transformer hidden layers
    nhead: int    # number of heads in attention
    attn_prob_dropout_ratio: float    # attention score dropout ratio
    hidden_dropout_ratio: float    # dropout ration before residual
    norm_first: bool    # norm_first
    fp16: bool    # fp16 presion


class MultiHeadAttention1DFunc(Function):

    @staticmethod
    def forward(ctx, input, input_mask, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias, norm_weight,
                norm_bias, config):
        cuda_module = colossal_multihead_attention
        forward_func = (cuda_module.multihead_attention_fw_fp16
                        if config.fp16 else cuda_module.multihead_attention_fw_fp32)
        if config.fp16:
            input = input.to(torch.half)
            input_mask = input_mask.to(torch.half)

        (output,) = forward_func(config.layer_id, input, input_mask, in_proj_weight, in_proj_bias, out_proj_weight,
                                 out_proj_bias, norm_weight, norm_bias, config.training, config.norm_first)

        if config.is_grad_enabled and config.training:
            ctx.save_for_backward(output, input, input_mask, in_proj_weight, in_proj_bias, out_proj_weight,
                                  out_proj_bias, norm_weight, norm_bias)
            ctx.config = config
        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert ctx.config.training

        cuda_module = colossal_multihead_attention
        backward_func = (cuda_module.multihead_attention_bw_fp16
                         if ctx.config.fp16 else cuda_module.multihead_attention_bw_fp32)

        output, input, input_mask, in_proj_weight, in_proj_bias, out_proj_weight, \
            out_proj_bias, norm_weight, norm_bias = ctx.saved_tensors

        grad_input = None
        grad_in_proj_weight = None
        grad_in_proj_bias = None
        grad_out_proj_weight = None
        grad_out_proj_bias = None
        grad_norm_weight = None
        grad_norm_bias = None

        if ctx.config.fp16:
            grad_output = grad_output.to(torch.half)
            output = output.to(torch.half)
            input = input.to(torch.half)
            input_mask = input_mask.to(torch.half)
        grad_input, grad_in_proj_weight, grad_in_proj_bias, grad_out_proj_weight, \
            grad_out_proj_bias, grad_norm_weight, grad_norm_bias = backward_func(
                ctx.config.layer_id, grad_output, output, input, input_mask, in_proj_weight,
                in_proj_bias, out_proj_weight, out_proj_bias, norm_weight, norm_bias)

        return (grad_input, None, grad_in_proj_weight, grad_in_proj_bias, grad_out_proj_weight, grad_out_proj_bias,
                grad_norm_weight, grad_norm_bias, None)


class MultiHeadAttention(nn.Module):
    """Initialize the MultiHeadAttention.

    Static variable:

        layer_id: The layer-index counter starting from 0 and incrementing by 1 every time a layer object is instantiated,
        e.g. if a model has 24 transformer layers, layer_id goes from 0 to 23.

    Arguments:
        hidden_size: Total dimension of hidden_size.
        nhead: Number of parallel attention heads.
        batch_size: Batch Size for one foward
        max_seq_len: Max length of input sequence
        dropout: Dropout probability
        norm_first: perform LayerNorms before attention
    """

    layer_id = 0

    def __init__(self, hidden_size, nhead, batch_size, max_seq_len, dropout=0.0, norm_first=False, fp16=True, pg=None):
        super(MultiHeadAttention, self).__init__()

        self.config = Config(batch_size * max_seq_len, max_seq_len, hidden_size, nhead, dropout, dropout, norm_first,
                             fp16)
        check_config(self.config)
        self.pg = pg
        self.pg_size = 1
        if self.pg:
            self.pg_size = pg.size()
        self.config.layer_id = MultiHeadAttention.layer_id
        MultiHeadAttention.layer_id = MultiHeadAttention.layer_id + 1

        # Load cuda modules if needed
        global colossal_multihead_attention
        if colossal_multihead_attention is None:
            from colossalai.kernel.op_builder import MultiHeadAttnBuilder
            multihead_attention = MultiHeadAttnBuilder().load()
            colossal_multihead_attention = multihead_attention

        # create the layer in cuda kernels.
        cuda_module = colossal_multihead_attention
        create_layer_func = (cuda_module.create_multihead_attention_fp16
                             if self.config.fp16 else cuda_module.create_multihead_attention_fp32)

        create_layer_func(
            self.config.layer_id,
            self.config.max_batch_tokens,
            self.config.max_seq_len,
            self.config.hidden_size,
            self.config.nhead,
            self.config.attn_prob_dropout_ratio,
            self.config.hidden_dropout_ratio,
            self.config.norm_first,
            self.pg,
        )

        hs = self.config.hidden_size

        self.precision = torch.float32
        if self.config.fp16:
            self.precision = torch.half

        self.hs_per_rank = int(hs / self.pg_size)

        self.in_proj_weight = nn.Parameter(torch.Tensor(3, self.hs_per_rank, hs))
        self.in_proj_bias = nn.Parameter(torch.Tensor(3, self.hs_per_rank))
        self.out_proj_weight = nn.Parameter(torch.Tensor(hs, self.hs_per_rank))
        self.out_proj_bias = nn.Parameter(torch.Tensor(hs))
        self.norm_weight = nn.Parameter(torch.Tensor(hs))
        self.norm_bias = nn.Parameter(torch.Tensor(hs))

        self.reset_parameters()
        torch.cuda.empty_cache()

    def calc_bound(self, w):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
        bound = 1.0 / math.sqrt(fan_in)
        return bound

    def reset_parameters(self):
        hs = self.config.hidden_size

        nn.init.zeros_(self.out_proj_bias)

        nn.init.ones_(self.norm_weight)
        nn.init.zeros_(self.norm_bias)

        if self.pg_size > 1:
            rank_in_pg = torch.distributed.get_rank(self.pg)
            attn_qkvw_global = torch.empty(hs * 3, hs)
            attn_qkvb_global = torch.empty(hs * 3)
            nn.init.xavier_uniform_(attn_qkvw_global, 1.0 / math.sqrt(2.0))
            bound = self.calc_bound(attn_qkvw_global)
            nn.init.uniform_(attn_qkvb_global, -bound, bound)

            attn_qkvw_global = attn_qkvw_global.cuda()
            attn_qkvb_global = attn_qkvb_global.cuda()
            torch.distributed.broadcast(attn_qkvw_global, src=0, group=self.pg)
            torch.distributed.broadcast(attn_qkvb_global, src=0, group=self.pg)
            attn_qkvw_global = attn_qkvw_global.cpu()
            attn_qkvb_global = attn_qkvb_global.cpu()

            with torch.no_grad():
                self.in_proj_weight.copy_(
                    attn_qkvw_global.view(3, hs, hs)[:,
                                                     int(hs * rank_in_pg / self.pg_size):int(hs * (rank_in_pg + 1) /
                                                                                             self.pg_size), :])
                self.in_proj_bias.copy_(
                    attn_qkvb_global.view(3, hs)[:,
                                                 int(hs * rank_in_pg / self.pg_size):int(hs * (rank_in_pg + 1) /
                                                                                         self.pg_size)])

            attn_ow_global = torch.empty(hs, hs)
            nn.init.xavier_uniform_(attn_ow_global, 1.0)
            attn_ow_global = attn_ow_global.cuda()
            torch.distributed.broadcast(attn_ow_global, src=0, group=self.pg)
            attn_ow_global = attn_ow_global.cpu()
            with torch.no_grad():
                self.out_proj_weight.copy_(attn_ow_global[:,
                                                          int(hs * rank_in_pg /
                                                              self.pg_size):int(hs * (rank_in_pg + 1) / self.pg_size)])

        else:
            attn_qkvw = self.in_proj_weight.view(-1, hs)
            nn.init.xavier_uniform_(attn_qkvw, 1.0 / math.sqrt(2.0))
            bound = self.calc_bound(attn_qkvw)
            nn.init.uniform_(self.in_proj_bias, -bound, bound)

            nn.init.xavier_uniform_(self.out_proj_weight, 1.0)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        destination = torch.nn.Module.state_dict(self, destination=destination, prefix=prefix, keep_vars=keep_vars)
        return destination

    def forward(self, hidden_states, encoder_padding_mask):
        self.config.training = self.training
        self.config.is_grad_enabled = torch.is_grad_enabled()
        hidden_states = hidden_states.contiguous()
        encoder_padding_mask = ((encoder_padding_mask * -1e8).type_as(hidden_states).contiguous())

        bs, sl, dim = hidden_states.size()
        if bs * sl > self.config.max_batch_tokens:
            raise ValueError(f"Batch token numbers {bs * sl} exceeds the limit {self.config.max_batch_tokens}.")
        if sl > self.config.max_seq_len:
            raise ValueError(f"Sequence length {sl} exceeds the limit {self.config.max_seq_len}.")
        if len(encoder_padding_mask.size()) == 1:
            assert bs == 1 and sl == encoder_padding_mask.size(0)
        else:
            assert bs == encoder_padding_mask.size(0) and sl == encoder_padding_mask.size(1)

        output = MultiHeadAttention1DFunc.apply(hidden_states, encoder_padding_mask, self.in_proj_weight,
                                                self.in_proj_bias, self.out_proj_weight, self.out_proj_bias,
                                                self.norm_weight, self.norm_bias, self.config)

        return output.to(self.precision)
