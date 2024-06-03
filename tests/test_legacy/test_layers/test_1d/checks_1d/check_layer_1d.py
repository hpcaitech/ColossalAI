import torch
import torch.distributed as dist
from torch.nn import Parameter

from colossalai.accelerator import get_accelerator
from colossalai.legacy.context.parallel_mode import ParallelMode
from colossalai.legacy.core import global_context as gpc
from colossalai.legacy.global_variables import tensor_parallel_env as env
from colossalai.legacy.nn import (
    Classifier1D,
    Embedding1D,
    Linear1D_Col,
    Linear1D_Row,
    VanillaClassifier,
    VocabParallelClassifier1D,
    VocabParallelCrossEntropyLoss1D,
    VocabParallelEmbedding1D,
)
from colossalai.legacy.utils import print_rank_0

from .common import BATCH_SIZE, DEPTH, HIDDEN_SIZE, NUM_CLASSES, SEQ_LENGTH, VOCAB_SIZE, check_equal


def check_linear_col():
    device = get_accelerator().get_current_device()
    dtype = torch.float32
    INPUT_SIZE = HIDDEN_SIZE
    OUTPUT_SIZE = 2 * HIDDEN_SIZE

    i = gpc.get_local_rank(ParallelMode.PARALLEL_1D)

    layer = Linear1D_Col(INPUT_SIZE, OUTPUT_SIZE)

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

    B_shape = OUTPUT_SIZE
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
    C = torch.chunk(C_master, DEPTH, dim=-1)[i]

    check_equal(out, C)
    print_rank_0("linear_col forward: pass")

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=get_accelerator().get_current_device())
    dist.broadcast(grad_master, src=0)
    grad = torch.chunk(grad_master, DEPTH, dim=-1)[i]
    grad = grad.clone()
    out.backward(grad)

    grad_master = grad_master.clone()
    C_master.backward(grad_master)
    A_grad = A_master.grad
    check_equal(A_grad, A.grad)

    W_grad = W_master.grad
    W_grad = torch.chunk(W_grad, DEPTH, dim=0)[i]
    check_equal(W_grad, layer.weight.grad)

    B_grad = B_master.grad
    B_grad = torch.chunk(B_grad, DEPTH, dim=0)[i]
    check_equal(B_grad, layer.bias.grad)

    print_rank_0("linear_col backward: pass")


def check_linear_row():
    device = get_accelerator().get_current_device()
    dtype = torch.float32
    INPUT_SIZE = HIDDEN_SIZE
    OUTPUT_SIZE = 2 * HIDDEN_SIZE

    i = gpc.get_local_rank(ParallelMode.PARALLEL_1D)

    layer = Linear1D_Row(OUTPUT_SIZE, INPUT_SIZE)

    A_shape = (BATCH_SIZE, SEQ_LENGTH, OUTPUT_SIZE)
    A_master = torch.randn(A_shape, dtype=dtype, device=device)
    dist.broadcast(A_master, src=0)
    A = torch.chunk(A_master, DEPTH, dim=-1)[i]
    A = A.clone()
    A.requires_grad = True

    W_shape = (INPUT_SIZE, OUTPUT_SIZE)
    W_master = torch.randn(W_shape, dtype=dtype, device=device)
    dist.broadcast(W_master, src=0)
    W = torch.chunk(W_master, DEPTH, dim=-1)[i]
    W = W.clone()
    W.requires_grad = True

    B_shape = INPUT_SIZE
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
    print_rank_0("linear_row forward: pass")

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=get_accelerator().get_current_device())
    dist.broadcast(grad_master, src=0)
    grad = grad_master.clone()
    out.backward(grad)

    grad_master = grad_master.clone()
    C_master.backward(grad_master)
    A_grad = A_master.grad
    A_grad = torch.chunk(A_grad, DEPTH, dim=-1)[i]
    check_equal(A_grad, A.grad)

    W_grad = W_master.grad
    W_grad = torch.chunk(W_grad, DEPTH, dim=-1)[i]
    check_equal(W_grad, layer.weight.grad)

    B_grad = B_master.grad
    check_equal(B_grad, layer.bias.grad)

    print_rank_0("linear_row backward: pass")


def check_embed():
    device = get_accelerator().get_current_device()
    dtype = torch.float32

    i = gpc.get_local_rank(ParallelMode.PARALLEL_1D)

    embed = Embedding1D(VOCAB_SIZE, HIDDEN_SIZE)
    embed = embed.to(dtype).to(device)
    embed_master = torch.nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
    embed_master = embed_master.to(dtype).to(device)

    weight_master = embed_master.weight.data
    torch.distributed.broadcast(weight_master, src=0)
    weight = torch.chunk(weight_master, DEPTH, dim=-1)[i]
    embed.weight.data.copy_(weight)

    A_shape = (BATCH_SIZE, SEQ_LENGTH)
    A_master = torch.randint(VOCAB_SIZE, A_shape, device=device)
    torch.distributed.broadcast(A_master, src=0)
    A = A_master.clone()
    out = embed(A)

    A_master = A_master.clone()
    C_master = embed_master(A_master)
    C = C_master.clone()
    check_equal(out, C)
    print_rank_0("embed forward: pass")

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(grad_master, src=0)
    grad = grad_master.clone()
    out.backward(grad)
    grad_master = grad_master.clone()
    C_master.backward(grad_master)

    B_grad = embed_master.weight.grad
    B_grad = torch.chunk(B_grad, DEPTH, dim=-1)[i]
    check_equal(B_grad, embed.weight.grad)
    print_rank_0("embed backward: pass")


def check_vocab_parallel_embed():
    device = get_accelerator().get_current_device()
    dtype = torch.float32

    i = gpc.get_local_rank(ParallelMode.PARALLEL_1D)

    embed = VocabParallelEmbedding1D(VOCAB_SIZE, HIDDEN_SIZE)
    embed = embed.to(dtype).to(device)
    embed_master = torch.nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
    embed_master = embed_master.to(dtype).to(device)

    weight_master = embed_master.weight.data
    torch.distributed.broadcast(weight_master, src=0)
    weight = torch.chunk(weight_master, DEPTH, dim=0)[i]
    embed.weight.data.copy_(weight)

    A_shape = (BATCH_SIZE, SEQ_LENGTH)
    A_master = torch.randint(VOCAB_SIZE, A_shape, device=device)
    torch.distributed.broadcast(A_master, src=0)
    A = A_master.clone()
    out = embed(A)

    A_master = A_master.clone()
    C_master = embed_master(A_master)
    C = C_master.clone()
    check_equal(out, C)
    print_rank_0("vocab parallel embed forward: pass")

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(grad_master, src=0)
    grad = grad_master.clone()
    out.backward(grad)
    grad_master = grad_master.clone()
    C_master.backward(grad_master)

    B_grad = embed_master.weight.grad
    B_grad = torch.chunk(B_grad, DEPTH, dim=0)[i]
    check_equal(B_grad, embed.weight.grad)
    print_rank_0("vocab parallel embed backward: pass")


def check_classifier_no_given_weight():
    device = get_accelerator().get_current_device()
    dtype = torch.float32

    i = gpc.get_local_rank(ParallelMode.PARALLEL_1D)

    env.parallel_input_1d = False
    parallel_input_1d = env.parallel_input_1d
    layer = Classifier1D(HIDDEN_SIZE, NUM_CLASSES, bias=True)
    layer.to(dtype).to(device)

    layer_master = VanillaClassifier(HIDDEN_SIZE, NUM_CLASSES, bias=True)
    layer_master = layer_master.to(dtype).to(device)

    W_master = layer_master.weight.data
    dist.broadcast(W_master, src=0)
    W = torch.chunk(W_master, DEPTH, dim=-1)[i]
    layer.weight.data.copy_(W)
    B_master = layer_master.bias.data
    dist.broadcast(B_master, src=0)
    B = B_master.clone()
    layer.bias.data.copy_(B)

    A_shape = (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE)
    A_master = torch.randn(A_shape, dtype=dtype, device=device)
    dist.broadcast(A_master, src=0)
    if parallel_input_1d:
        A = torch.chunk(A_master, DEPTH, dim=-1)[i]
        A = A.clone()
    else:
        A = A_master.clone()
    A.requires_grad = True

    out = layer(A)

    A_master = A_master.clone()
    A_master.requires_grad = True
    C_master = layer_master(A_master)
    C = C_master.clone()

    check_equal(out, C)
    print_rank_0("classifier (no given weight) forward: pass")

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=device)
    dist.broadcast(grad_master, src=0)
    grad = grad_master.clone()
    out.backward(grad)

    grad_master = grad_master.clone()
    C_master.backward(grad_master)
    A_grad = A_master.grad
    if parallel_input_1d:
        A_grad = torch.chunk(A_grad, DEPTH, dim=-1)[i]
    check_equal(A_grad, A.grad)

    W_grad = layer_master.weight.grad
    W_grad = torch.chunk(W_grad, DEPTH, dim=-1)[i]
    check_equal(W_grad, layer.weight.grad)

    B_grad = layer_master.bias.grad
    check_equal(B_grad, layer.bias.grad)

    print_rank_0("classifier (no given weight) backward: pass")


def check_vocab_parallel_classifier_no_given_weight():
    device = get_accelerator().get_current_device()
    dtype = torch.float32

    i = gpc.get_local_rank(ParallelMode.PARALLEL_1D)

    layer = VocabParallelClassifier1D(HIDDEN_SIZE, VOCAB_SIZE, bias=True)
    layer.to(dtype).to(device)

    layer_master = VanillaClassifier(HIDDEN_SIZE, VOCAB_SIZE, bias=True)
    layer_master = layer_master.to(dtype).to(device)

    W_master = layer_master.weight.data
    dist.broadcast(W_master, src=0)
    W = torch.chunk(W_master, DEPTH, dim=0)[i]
    layer.weight.data.copy_(W)
    B_master = layer_master.bias.data
    dist.broadcast(B_master, src=0)
    B = torch.chunk(B_master, DEPTH, dim=0)[i]
    layer.bias.data.copy_(B)

    A_shape = (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE)
    A_master = torch.randn(A_shape, dtype=dtype, device=device)
    dist.broadcast(A_master, src=0)
    A = A_master.clone()
    A.requires_grad = True

    out = layer(A)

    A_master = A_master.clone()
    A_master.requires_grad = True
    C_master = layer_master(A_master)
    C = torch.chunk(C_master, DEPTH, dim=-1)[i]

    check_equal(out, C)
    print_rank_0("vocab parallel classifier (no given weight) forward: pass")

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=device)
    dist.broadcast(grad_master, src=0)
    grad = torch.chunk(grad_master, DEPTH, dim=-1)[i]
    grad = grad.clone()
    out.backward(grad)

    grad_master = grad_master.clone()
    C_master.backward(grad_master)
    A_grad = A_master.grad
    check_equal(A_grad, A.grad)

    W_grad = layer_master.weight.grad
    W_grad = torch.chunk(W_grad, DEPTH, dim=0)[i]
    check_equal(W_grad, layer.weight.grad)

    B_grad = layer_master.bias.grad
    B_grad = torch.chunk(B_grad, DEPTH, dim=0)[i]
    check_equal(B_grad, layer.bias.grad)

    print_rank_0("vocab parallel classifier (no given weight) backward: pass")


def check_classifier_given_embed_weight():
    device = get_accelerator().get_current_device()
    dtype = torch.float32

    i = gpc.get_local_rank(ParallelMode.PARALLEL_1D)

    embed = Embedding1D(VOCAB_SIZE, HIDDEN_SIZE)
    embed = embed.to(dtype).to(device)
    embed_master = torch.nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
    embed_master = embed_master.to(dtype).to(device)

    weight_master = embed_master.weight.data
    torch.distributed.broadcast(weight_master, src=0)
    weight = torch.chunk(weight_master, DEPTH, dim=-1)[i]
    embed.weight.data.copy_(weight)

    env.parallel_input_1d = False
    layer = Classifier1D(HIDDEN_SIZE, NUM_CLASSES, weight=embed.weight, bias=False)
    layer.to(dtype).to(device)

    layer_master = VanillaClassifier(HIDDEN_SIZE, NUM_CLASSES, weight=embed_master.weight, bias=False)
    layer_master = layer_master.to(dtype).to(device)

    A_shape = (BATCH_SIZE, SEQ_LENGTH)
    A_master = torch.randint(VOCAB_SIZE, A_shape, device=device)
    torch.distributed.broadcast(A_master, src=0)
    A = A_master.clone()
    out = layer(embed(A))

    A_master = A_master.clone()
    C_master = layer_master(embed_master(A_master))
    C = C_master.clone()
    check_equal(out, C)
    print_rank_0("classifier (given embed weight) forward: pass")

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=device)
    dist.broadcast(grad_master, src=0)
    grad = grad_master.clone()
    out.backward(grad)

    grad_master = grad_master.clone()
    C_master.backward(grad_master)

    W_grad = embed_master.weight.grad
    W_grad = torch.chunk(W_grad, DEPTH, dim=-1)[i]
    check_equal(W_grad, embed.weight.grad)

    print_rank_0("classifier (given embed weight) backward: pass")


def check_vocab_parallel_classifier_given_embed_weight():
    device = get_accelerator().get_current_device()
    dtype = torch.float32

    i = gpc.get_local_rank(ParallelMode.PARALLEL_1D)

    embed = VocabParallelEmbedding1D(VOCAB_SIZE, HIDDEN_SIZE)
    embed = embed.to(dtype).to(device)
    embed_master = torch.nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
    embed_master = embed_master.to(dtype).to(device)

    weight_master = embed_master.weight.data
    torch.distributed.broadcast(weight_master, src=0)
    weight = torch.chunk(weight_master, DEPTH, dim=0)[i]
    embed.weight.data.copy_(weight)

    env.parallel_input_1d = False
    layer = VocabParallelClassifier1D(HIDDEN_SIZE, NUM_CLASSES, weight=embed.weight, bias=False)
    layer.to(dtype).to(device)

    layer_master = VanillaClassifier(HIDDEN_SIZE, NUM_CLASSES, weight=embed_master.weight, bias=False)
    layer_master = layer_master.to(dtype).to(device)

    A_shape = (BATCH_SIZE, SEQ_LENGTH)
    A_master = torch.randint(VOCAB_SIZE, A_shape, device=device)
    torch.distributed.broadcast(A_master, src=0)
    A = A_master.clone()
    out = layer(embed(A))

    A_master = A_master.clone()
    C_master = layer_master(embed_master(A_master))
    C = torch.chunk(C_master, DEPTH, dim=-1)[i]
    check_equal(out, C)
    print_rank_0("vocab parallel classifier (given embed weight) forward: pass")

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=device)
    dist.broadcast(grad_master, src=0)
    grad = torch.chunk(grad_master, DEPTH, dim=-1)[i]
    grad = grad.clone()
    out.backward(grad)

    grad_master = grad_master.clone()
    C_master.backward(grad_master)

    W_grad = embed_master.weight.grad
    W_grad = torch.chunk(W_grad, DEPTH, dim=0)[i]
    check_equal(W_grad, embed.weight.grad)

    print_rank_0("vocab parallel classifier (given embed weight) backward: pass")


def check_vocab_parallel_loss():
    device = get_accelerator().get_current_device()
    dtype = torch.float32

    i = gpc.get_local_rank(ParallelMode.PARALLEL_1D)

    criterion = VocabParallelCrossEntropyLoss1D()
    criterion_master = torch.nn.CrossEntropyLoss()

    out_shape = (BATCH_SIZE, SEQ_LENGTH, NUM_CLASSES)
    out_master = torch.randn(out_shape, dtype=dtype, device=device)
    target_master = torch.randint(NUM_CLASSES, (BATCH_SIZE, SEQ_LENGTH), dtype=torch.long, device=device)
    torch.distributed.broadcast(out_master, src=0)
    torch.distributed.broadcast(target_master, src=0)
    out = torch.chunk(out_master, DEPTH, dim=-1)[i]
    out = out.clone()
    out.requires_grad = True

    loss = criterion(out, target_master)

    out_master = out_master.clone()
    out_master.requires_grad = True
    loss_master = criterion_master(out_master, target_master)
    check_equal(loss, loss_master)
    print_rank_0("vocab parallel loss forward: pass")

    loss.backward()
    loss_master.backward()

    out_grad = out_master.grad
    out_grad = torch.chunk(out_grad, DEPTH, dim=-1)[i]
    check_equal(out_grad, out.grad)
    print_rank_0("vocab parallel loss backward: pass")


@torch.no_grad()
def check_linear_row_stream_inference():
    device = get_accelerator().get_current_device()
    dtype = torch.float32
    INPUT_SIZE = HIDDEN_SIZE
    OUTPUT_SIZE = 2 * HIDDEN_SIZE

    i = gpc.get_local_rank(ParallelMode.PARALLEL_1D)

    stream_chunk_num = 4
    assert HIDDEN_SIZE % stream_chunk_num == 0
    layer = Linear1D_Row(OUTPUT_SIZE, INPUT_SIZE, stream_chunk_num=stream_chunk_num)

    A_shape = (BATCH_SIZE, SEQ_LENGTH, OUTPUT_SIZE)
    A_master = torch.randn(A_shape, dtype=dtype, device=device)
    dist.broadcast(A_master, src=0)
    A = torch.chunk(A_master, DEPTH, dim=-1)[i]
    A = A.clone()

    W_shape = (INPUT_SIZE, OUTPUT_SIZE)
    W_master = torch.randn(W_shape, dtype=dtype, device=device)
    dist.broadcast(W_master, src=0)
    W = torch.chunk(W_master, DEPTH, dim=-1)[i]
    W = W.clone()

    B_shape = INPUT_SIZE
    B_master = torch.randn(B_shape, dtype=dtype, device=device)
    dist.broadcast(B_master, src=0)
    B = B_master.clone()

    layer.weight = Parameter(W)
    layer.bias = Parameter(B)
    layer.chunk_weight()
    layer.eval()

    out = layer(A)

    A_master = A_master.clone()
    W_master = W_master.clone()
    B_master = B_master.clone()
    C_master = torch.matmul(A_master, W_master.transpose(0, 1)) + B_master
    C = C_master.clone()

    check_equal(out, C)
    print_rank_0("linear_row forward: pass")
