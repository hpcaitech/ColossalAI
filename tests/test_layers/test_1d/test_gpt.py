from colossalai.nn.optimizer.zero_redundancy_optimizer_level_2 import print_rank_msg
import torch
import torch.distributed as dist
from torch.nn import Parameter
from colossalai.logging import get_global_dist_logger
import time
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.nn import Linear1D_Col, Linear1D_Row, GPTSelfAttention1D, GPTEmbedding1D, GPTLMHead1D, GPTTransformerLayer1D, ViTMLP1D, ViTSelfAttention1D, ViTPatchEmbedding1D, ViTHead1D, ViTTokenFuser1D
from colossalai.utils import get_current_device, print_rank_0
from common import HIDDEN_SIZE, DEPTH, BATCH_SIZE, SEQ_LENGTH, NUM_CLASSES, check_equal, IMG_SIZE


def check_attention():
    device = get_current_device()
    dtype = torch.float32
    INPUT_SIZE = HIDDEN_SIZE
    NUM_ATTENTION_HEADS = 2

    i = gpc.get_local_rank(ParallelMode.PARALLEL_1D)

    layer = GPTSelfAttention1D(
        HIDDEN_SIZE, 
        NUM_ATTENTION_HEADS, 
        0.5,
        0.5,
        max_position_embeddings = 10,
    ).to(device=device)

    A_shape = (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)
    A_master = torch.randn(A_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(A_master, src=0)
    A = A_master.clone()
    A.requires_grad = True

    mask_shape = (BATCH_SIZE, NUM_ATTENTION_HEADS // DEPTH, SEQ_LENGTH, SEQ_LENGTH)
    attention_mask = torch.zeros(mask_shape, dtype=dtype, device=device)

    out = layer(A)
    assert out.shape == (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)
    print_rank_0('self attention forward: pass')

    grad_shape = out.shape
    grad = torch.randn(grad_shape, dtype=dtype, device=device)

    out.backward(grad)
    assert A.grad.shape == A.shape
    print_rank_0('self attention backward: pass')

def check_embedding():
    device = get_current_device()
    dtype = torch.float32
    logger = get_global_dist_logger()
    rank = torch.distributed.get_rank()

    i = gpc.get_local_rank(ParallelMode.PARALLEL_1D)

    layer_master = TestGPTembed(
        embed_dim = 16,
        vocab_size = 10,
        max_position_embeddings=10,
        dropout_embed = 0,
    ).to(device=device)

    layer = GPTEmbedding1D(
        embed_dim = 16,
        vocab_size = 10,
        max_position_embeddings=10,
        dropout_embed = 0,
    ).to(device=device)
        
    A_shape = (BATCH_SIZE, SEQ_LENGTH)
    A_master = torch.randint(0, 10, A_shape, device=device)
    torch.distributed.broadcast(A_master, src=0)
    A = A_master.clone()
    # A.requires_grad = True
    A_master = A_master.clone()
    out = layer(A)

    layer_master.wte.weight = layer.wte.weight
    layer_master.wpe.weight = layer.wpe.weight
    C = layer_master(A_master)
    
    #check seed
    print('layer.wte.weight identical?', layer.wte.weight)
    print('layer.wpe.weight identical?', layer.wte.weight)

    print('output size: ',out.size())
    assert out.shape == (BATCH_SIZE, SEQ_LENGTH, 16)
    print_rank_0('embedding forward: pass')
    
    logger.info('Rank {} embed forward output: {}'.format(
        rank, check_equal(C, out)))

    grad_shape = out.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(grad_master, src=0)
    grad = grad_master.clone()
    grad_master = grad_master.clone()

    C.backward(grad_master)
    out.backward(grad)

    assert layer.wte.grad.shape == layer.wte.shape
    print_rank_0('embedding backward: pass')

    logger.info('Rank {} embed backward (wte.grad): {}'.format(
        rank, check_equal(layer.wte.grad, layer_master.wte.grad)))

    logger.info('Rank {} embed backward (wpe.grad): {}'.format(
        rank, check_equal(layer.wpe.grad, layer_master.wpe.grad)))

class TestGPTembed(torch.nn.Module):
    def __init__(self,
                 embed_dim,
                 vocab_size,
                 max_position_embeddings,
                 dropout_embed,
                 weight_init='torch'):
        super().__init__()

        self.embed_dim = embed_dim
        self.dropout_embed = torch.nn.Dropout(dropout_embed)
        self.wte = torch.nn.Embedding(vocab_size, self.embed_dim)
        self.wpe = torch.nn.Embedding(max_position_embeddings, self.embed_dim)

    def forward(self, input_ids = None, input_embeds = None, position_ids = None, token_type_ids = None):
        assert input_ids is not None or input_embeds is not None
        if input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif input_embeds is not None:
            input_shape = input_embeds.size()[:-1]

        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if position_ids is None:
            position_ids = torch.arange(0, input_shape[-1] + 0, dtype=torch.long, device=get_current_device())
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        if input_embeds is None:
            input_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = input_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.dropout_embed(hidden_states)
        return hidden_states


