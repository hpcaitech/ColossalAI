import torch
import torch.distributed as dist
from torch.nn import Parameter
import time
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.nn import Linear1D_Col, Linear1D_Row, TransformerMLP1D, TransformerSelfAttention1D, ViTMLP1D, ViTSelfAttention1D, ViTPatchEmbedding1D, ViTHead1D, ViTTokenFuser1D
from colossalai.utils import get_current_device, print_rank_0
from .common import HIDDEN_SIZE, DEPTH, BATCH_SIZE, SEQ_LENGTH, NUM_CLASSES, check_equal, IMG_SIZE


def check_linear_col():
    device = get_current_device()
    dtype = torch.float32
    INPUT_SIZE = HIDDEN_SIZE
    OUTPUT_SIZE = 2 * HIDDEN_SIZE

    i = gpc.get_local_rank(ParallelMode.PARALLEL_1D)

    layer = Linear1D_Col(INPUT_SIZE, OUTPUT_SIZE, gather_output=True)

    A_shape = (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)
    A_master = torch.randn(A_shape, dtype=dtype, device=device)
    dist.broadcast(A_master, src=0)
    A = A_master.clone()
    A.requires_grad = True

    W_shape = (OUTPUT_SIZE, INPUT_SIZE)
    W_master = torch.randn(W_shape, dtype=dtype, device=device)
    dist.broadcast(W_master, src=0)
    W = torch.chunk(W_master, DEPTH, dim=0)[i]
    W = W.clone()
    W.requires_grad = True

    B_shape = (OUTPUT_SIZE)
    B_master = torch.randn(B_shape, dtype=dtype, device=device)
    dist.broadcast(B_master, src=0)
    B = torch.chunk(B_master, DEPTH, dim=0)[i]
    B = B.clone()
    B.requires_grad = True

    layer.weight = Parameter(W)
    layer.bias = Parameter(B)
    out = layer(A)

    A_master = A_master.clone()
    A_master.requires_grad = True
    W_master = W_master.clone()
    W_master.requires_grad = True
    B_master = B_master.clone()
    B_master.requires_grad = True
    C_master = torch.matmul(A_master, W_master.transpose(0, 1)) + B_master
    C = C_master.clone()

    check_equal(out, C)
    print_rank_0('linear_col gather_output forward: pass')

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=get_current_device())
    dist.broadcast(grad_master, src=0)
    grad = grad_master.detach()
    out.backward(grad)

    C_master.backward(grad)
    A_grad = A_master.grad
    check_equal(A_grad, A.grad)

    W_grad = W_master.grad
    W_grad = torch.chunk(W_grad, DEPTH, dim=0)[i]
    check_equal(W_grad, layer.weight.grad)

    B_grad = B_master.grad
    B_grad = torch.chunk(B_grad, DEPTH, dim=0)[i]
    check_equal(B_grad, layer.bias.grad)

    print_rank_0('linear_col gather_output backward: pass')


def check_linear_row():
    device = get_current_device()
    dtype = torch.float32
    INPUT_SIZE = HIDDEN_SIZE
    OUTPUT_SIZE = 2 * HIDDEN_SIZE

    i = gpc.get_local_rank(ParallelMode.PARALLEL_1D)

    layer = Linear1D_Row(OUTPUT_SIZE, INPUT_SIZE, parallel_input=False)

    A_shape = (BATCH_SIZE, SEQ_LENGTH, OUTPUT_SIZE)
    A_master = torch.randn(A_shape, dtype=dtype, device=device)
    dist.broadcast(A_master, src=0)
    A = A_master.clone()
    A.requires_grad = True

    W_shape = (INPUT_SIZE, OUTPUT_SIZE)
    W_master = torch.randn(W_shape, dtype=dtype, device=device)
    dist.broadcast(W_master, src=0)
    W = torch.chunk(W_master, DEPTH, dim=-1)[i]
    W = W.clone()
    W.requires_grad = True

    B_shape = (INPUT_SIZE)
    B_master = torch.randn(B_shape, dtype=dtype, device=device)
    dist.broadcast(B_master, src=0)
    B = B_master.clone()
    B.requires_grad = True

    layer.weight = Parameter(W)
    layer.bias = Parameter(B)
    out = layer(A)

    A_master = A_master.clone()
    A_master.requires_grad = True
    W_master = W_master.clone()
    W_master.requires_grad = True
    B_master = B_master.clone()
    B_master.requires_grad = True
    C_master = torch.matmul(A_master, W_master.transpose(0, 1)) + B_master
    C = C_master.clone()

    check_equal(out, C)
    print_rank_0('linear_row no parallel_input forward: pass')

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=get_current_device())
    dist.broadcast(grad_master, src=0)
    grad = grad_master.detach()
    out.backward(grad)

    C_master.backward(grad)
    A_grad = A_master.grad
    check_equal(A_grad, A.grad)

    W_grad = W_master.grad
    W_grad = torch.chunk(W_grad, DEPTH, dim=-1)[i]
    check_equal(W_grad, layer.weight.grad)

    B_grad = B_master.grad
    check_equal(B_grad, layer.bias.grad)

    print_rank_0('linear_row no parallel_input backward: pass')


class Testvithead(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        x = x[:, 0]
        x = self.linear(x)
        return x


def check_head():
    device = get_current_device()
    dtype = torch.float32
    INPUT_SIZE = HIDDEN_SIZE

    i = gpc.get_local_rank(ParallelMode.PARALLEL_1D)

    head = ViTHead1D(INPUT_SIZE, NUM_CLASSES, dtype=dtype)
    torch.nn.init.zeros_(head.linear.bias)
    torch.nn.init.ones_(head.linear.weight)
    head = head.to(device)

    layer = Testvithead(INPUT_SIZE, NUM_CLASSES, bias=True)
    torch.nn.init.zeros_(layer.linear.bias)
    torch.nn.init.ones_(layer.linear.weight)
    layer = layer.to(device)

    A_shape = (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)
    A_master = torch.randn(A_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(A_master, src=0)
    A = A_master.clone()
    A.requires_grad = True

    fwd_start = time.time()
    out = head(A)
    fwd_end = time.time()
    print_rank_0(
        'head forward: pass | {0} --> {1} | {2:.3f} s'.format(
            tuple(A.shape), tuple(out.shape), fwd_end - fwd_start))
    A_master = A_master.clone()
    A_master.requires_grad = True
    C_master = layer(A_master)
    # C = torch.chunk(C_master, DEPTH, dim=0)[i]
    print_rank_0('Rank {} head forward: {}'.format(i, check_equal(out, C_master)))

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape,
                              dtype=dtype,
                              device=get_current_device())
    torch.distributed.broadcast(grad_master, src=0)
    # grad = torch.chunk(grad_master, DEPTH, dim=0)[i]

    # bwd_start = time.time()
    out.backward(grad_master)
    # bwd_end = time.time()
    # print_rank_0('head backward: pass | {:.3f} s'.format(bwd_end - bwd_start),
    #  logger)

    C_master.backward(grad_master)
    A_grad = A_master.grad
    # if j == 0:
    print_rank_0('Rank {} head backward (input_grad): {}'.format(
        i, check_equal(A_grad, A.grad)))


class Testvitembed(torch.nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_chans: int,
                 embed_size: int, drop_prob: float) -> None:
        super().__init__()
        self.proj = torch.nn.Conv2d(in_chans,
                                    embed_size,
                                    kernel_size=patch_size,
                                    stride=patch_size)
        num_patches = (img_size // patch_size)**2
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, embed_size))
        self.pos_embed = torch.nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_size))
        self.pos_drop = torch.nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        return x


def check_embed():
    device = get_current_device()
    dtype = torch.float32
    i = gpc.get_local_rank(ParallelMode.PARALLEL_1D)

    layer = ViTPatchEmbedding1D(IMG_SIZE, 4, HIDDEN_SIZE)
    layer2 = ViTTokenFuser1D(IMG_SIZE, 4, HIDDEN_SIZE)
    torch.nn.init.zeros_(layer.proj.bias)
    torch.nn.init.ones_(layer.proj.weight)
    torch.nn.init.ones_(layer2.cls_token)
    torch.nn.init.ones_(layer2.pos_embed)
    layer = layer.to(device)
    layer2 = layer2.to(device)

    layer_master = Testvitembed(IMG_SIZE, 4, 3, HIDDEN_SIZE, 0.)
    torch.nn.init.zeros_(layer_master.proj.bias)
    torch.nn.init.ones_(layer_master.proj.weight)
    torch.nn.init.ones_(layer_master.cls_token)
    torch.nn.init.ones_(layer_master.pos_embed)
    layer_master = layer_master.to(device)

    A_shape = (BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
    A_master = torch.randn(A_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(A_master, src=0)
    A = A_master.clone()
    A.requires_grad = True

    fwd_start = time.time()
    out = layer2(layer(A))
    fwd_end = time.time()
    print_rank_0(
        'embedding forward: pass | {0} --> {1} | {2:.3f} s'.format(
            tuple(A.shape), tuple(out.shape), fwd_end - fwd_start))
    # out_cls = out[:, 0]
    # out_tensor = out[:, 1:]

    A_master = A_master.clone()
    A_master.requires_grad = True
    C_master = layer_master(A_master)
    # if j == 0:
    #     C_cls = C_master[:, 0]
    #     C_cls = torch.chunk(C_cls, DEPTH, dim=0)[i]
    #     C_cls = torch.chunk(C_cls, DEPTH, dim=-1)[k]
    #     logger.info('Rank {} embed forward (cls): {}'.format(
    #         rank, check_equal(out_cls, C_cls)))
    # C = C_master[:, 1:]
    print_rank_0('Rank {} embed forward: {}'.format(i, check_equal(out, C_master)))

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape,
                              dtype=dtype,
                              device=get_current_device())
    torch.distributed.broadcast(grad_master, src=0)
    # cls_grad = grad_master[:, 0]
    # cls_grad = torch.chunk(cls_grad, DEPTH, dim=0)[i]
    # cls_grad = torch.chunk(cls_grad, DEPTH, dim=-1)[k]
    # grad = grad_master[:, 1:]
    # grad = torch.cat((torch.unsqueeze(cls_grad, 1), grad), dim=1)
    bwd_start = time.time()
    out.backward(grad_master)
    bwd_end = time.time()
    print_rank_0(
        'embedding backward: pass | {:.3f} s'.format(bwd_end - bwd_start))

    C_master.backward(grad_master)

    A_grad = A_master.grad
    print_rank_0('Rank {} embed backward (input_grad): {}'.format(i, check_equal(A_grad, A.grad)))

    print_rank_0('Rank {} embed backward (cls_grad): {}'.format(
        i, check_equal(layer_master.cls_token.grad, layer2.cls_token.grad)))

    print_rank_0('Rank {} embed backward (pos_embed_grad): {}'.format(
        i, check_equal(layer_master.pos_embed.grad, layer2.pos_embed.grad)))

    print_rank_0('Rank {} embed backward (proj_weight_grad): {}'.format(
        i, check_equal(layer_master.proj.weight.grad, layer.proj.weight.grad)))

    print_rank_0('Rank {} embed backward (proj_bias_grad): {}'.format(
        i, check_equal(layer_master.proj.bias.grad, layer.proj.bias.grad)))

    return fwd_end - fwd_start, bwd_end - bwd_start


def check_attention():
    device = get_current_device()
    dtype = torch.float32
    INPUT_SIZE = HIDDEN_SIZE
    NUM_ATTENTION_HEADS = 2

    i = gpc.get_local_rank(ParallelMode.PARALLEL_1D)

    layer = ViTSelfAttention1D(
        HIDDEN_SIZE,
        NUM_ATTENTION_HEADS,
        0.5,
        0.5
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


def check_mlp():
    device = get_current_device()
    dtype = torch.float32
    INPUT_SIZE = HIDDEN_SIZE

    i = gpc.get_local_rank(ParallelMode.PARALLEL_1D)

    layer = ViTMLP1D(
        HIDDEN_SIZE,
        4.0
    ).to(device=device)

    A_shape = (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)
    A_master = torch.randn(A_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(A_master, src=0)
    A = A_master.clone()
    A.requires_grad = True

    out = layer(A)
    assert out.shape == (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)
    print_rank_0('mlp forward: pass')

    grad_shape = out.shape
    grad = torch.randn(grad_shape, dtype=dtype, device=device)

    out.backward(grad)
    assert A.grad.shape == A.shape
    print_rank_0('mlp backward: pass')


def check_patch_embedding():
    device = get_current_device()
    dtype = torch.float32
    INPUT_SIZE = 4
    PATCH_SIZE = 2

    i = gpc.get_local_rank(ParallelMode.PARALLEL_1D)

    layer = ViTPatchEmbedding1D(
        INPUT_SIZE,
        PATCH_SIZE,
        HIDDEN_SIZE,
    ).to(device=device)

    A_shape = (BATCH_SIZE, 3, INPUT_SIZE, INPUT_SIZE)
    A_master = torch.randn(A_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(A_master, src=0)
    A = A_master.clone()
    A.requires_grad = True

    out = layer(A)
    print('output size: ', out.size())
    assert out.shape == (BATCH_SIZE, 4, HIDDEN_SIZE)
    print_rank_0('patch embedding forward: pass')

    grad_shape = out.shape
    grad = torch.randn(grad_shape, dtype=dtype, device=device)

    out.backward(grad)
    assert A.grad.shape == A.shape
    print_rank_0('patch embedding backward: pass')
