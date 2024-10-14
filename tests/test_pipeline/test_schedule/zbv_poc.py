import gc
import time
from copy import deepcopy

import torch
import torch.nn as nn
from torch.testing import assert_close


def get_model_numel(model):
    return sum(p.numel() for p in model.parameters()) / 1024**2


# Step1: dx = w*dy
def backward_b(loss, x, model):
    torch.autograd.backward(loss, inputs=x, retain_graph=True)


# Step2: dummy dw = x*dy
def backward_w(loss, model):
    torch.autograd.backward(loss, inputs=list(model.parameters()))


def test_double_dx_dw_split_nsync():
    device = "cuda:0"
    model = nn.Linear(4096, 4096, bias=None).to(device=device)
    # print(f"model numel {get_model_numel(model)}") # 4GB
    x1 = torch.rand(4096, 4096).to(device=device)
    x2 = torch.rand(4096, 4096).to(device=device)
    ref_model = deepcopy(model)
    ref_x1 = x1.clone()
    ref_x2 = x1.clone()

    # first step
    x1.requires_grad_()
    x2.requires_grad_()
    ref_x1.requires_grad_()
    ref_x2.requires_grad_()

    # loss for dx_dw bwd
    loss1 = model(x1).sum()
    loss2 = model(x2).sum()

    # loss for common bwd
    ref_loss1 = ref_model(ref_x1).sum()
    ref_loss2 = ref_model(ref_x2).sum()

    # dx1
    torch.cuda.synchronize()
    bwd_b_start_time = time.time()
    backward_b(loss1, x1, model)
    bwd_b_end_time = time.time()
    print(f"loss_1 bwd B runtime {bwd_b_end_time - bwd_b_start_time}")

    for p in model.parameters():
        assert p.grad is None
    assert x1.grad is not None

    # dx2
    torch.cuda.synchronize()
    bwd_b_start_time = time.time()
    backward_b(loss2, x2, model)
    bwd_b_end_time = time.time()
    print(f"loss_2 bwd B runtime {bwd_b_end_time - bwd_b_start_time}")

    # dw1
    torch.cuda.synchronize()
    bwd_w_start_time = time.time()
    backward_w(loss1, model)
    bwd_w_end_time = time.time()
    print(f"loss_1 bwd W runtime {bwd_w_end_time - bwd_w_start_time}")
    for p in model.parameters():
        assert p.grad is not None

    # common bwd 1
    torch.cuda.synchronize()
    comm_bwd_start_time = time.time()
    ref_loss1.backward()
    comm_bwd_end_time = time.time()
    print(f"loss_1 comm bwd runtime {comm_bwd_end_time - comm_bwd_start_time}")

    # # assert dx1 & dw1 == bwd 1
    # assert_close(x1.grad, ref_x1.grad)
    # for p1, p2 in zip(model.parameters(), ref_model.parameters()):
    #     assert_close(p1, p2)
    #     assert_close(p1.grad, p2.grad)

    # dw2
    torch.cuda.synchronize()
    bwd_w_start_time = time.time()
    backward_w(loss2, model)
    bwd_w_end_time = time.time()
    print(f"loss_2 bwd W runtime {bwd_w_end_time - bwd_w_start_time}")

    # common bwd 2
    torch.cuda.synchronize()
    comm_bwd_start_time = time.time()
    ref_loss2.backward()
    comm_bwd_end_time = time.time()
    print(f"loss_2 comm bwd runtime {comm_bwd_end_time - comm_bwd_start_time}")

    # # assert dx2 & dw2 == bwd 2
    # assert_close(x2.grad, ref_x2.grad)
    # for p1, p2 in zip(model.parameters(), ref_model.parameters()):
    #     print(f"bwd2:\n p1 {p1.grad},\n p2 {p2.grad}\n")
    #     assert_close(p1, p2)
    #     assert_close(p1.grad, p2.grad)


def test_double_dx_dw_split_sync():
    device = "cuda:0"
    model = nn.Linear(8, 8, bias=None).to(device=device)
    print(f"model size {get_model_numel(model)} ")  # 4GB
    x1 = torch.rand(8, 8).to(device=device)
    x2 = torch.rand(8, 8).to(device=device)

    # x1 = torch.ones(8, 8).to(device=device)
    # x2 = torch.ones(8, 8).to(device=device)

    ref_model = deepcopy(model)
    ref_x1 = x1.clone()
    ref_x2 = x2.clone()

    x1.requires_grad_()
    x2.requires_grad_()
    ref_x1.requires_grad_()
    ref_x2.requires_grad_()

    ############
    # step1:
    ############

    # loss1
    loss1 = model(x1).sum()

    # ref_loss1
    ref_model(ref_x1).sum()

    # dx1
    backward_b(loss1, x1, model)
    for p in model.parameters():
        assert p.grad is None
    assert x1.grad is not None

    # dw1
    backward_w(loss1, model)
    for p in model.parameters():
        assert p.grad is not None

    # common bwd 1
    # ref_loss1.backward()

    # assert dx1 & dw1 == bwd 1
    assert_close(x1.grad, ref_x1.grad)
    for p1, p2 in zip(model.parameters(), ref_model.parameters()):
        assert_close(p1, p2)
        assert_close(p1.grad, p2.grad)

    ############
    # step2:
    ############

    # loss2
    loss2 = model(x2).sum()

    # ref_loss2
    ref_loss2 = ref_model(ref_x2).sum()

    for p1, p2 in zip(model.parameters(), ref_model.parameters()):
        print(f"bwd2:\n p1 {p1.grad},\n p2 {p2.grad}\n")
        assert_close(p1, p2)
        assert_close(p1.grad, p2.grad)

    # dx2
    backward_b(loss2, x2, model)

    # dw2
    backward_w(loss2, model)

    # common bwd 2
    ref_loss2.backward()

    # assert dx2 & dw2 == bwd 2
    assert_close(x2.grad, ref_x2.grad)
    for p1, p2 in zip(model.parameters(), ref_model.parameters()):
        print(f"bwd2:\n p1 {p1.grad},\n p2 {p2.grad}\n")
        assert_close(p1, p2)
        assert_close(p1.grad, p2.grad)


def deallocate_output_tensor(out):
    """Pseudo-deallocate (i.e., set to scalar) the output tensor's '.data' field.

    This method should be called right after the output tensor has been
    sent to the next pipeline stage. At this point, the output tensor is
    only useful for its '.grad_fn' field, and not its '.data'.
    """
    assert isinstance(out, torch.Tensor), "expected Tensor, found %s." % type(out).__name__
    assert out._base is None, "counter-productive to free a view of another tensor."
    out.data = torch.empty(
        (1,),
        device=out.device,
        dtype=out.dtype,
    )


IN_DIM = 8192
OUT_DIM = 8192
NUM_LAYER = 3


class MlpModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(IN_DIM, OUT_DIM, bias=None) for _ in range(NUM_LAYER)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q, k, v = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


def mem_dx_dw():
    device = "cuda:0"
    # model = nn.Linear(IN_DIM, OUT_DIM, bias=None).to(device=device)
    print(f"Before init Model: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")
    model = MlpModel().to(device=device)
    print(f"model numel {get_model_numel(model)}")  # 4GB
    print(f"After init Model: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")

    print(f"Before init x1&2&3: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")
    x1 = torch.rand(IN_DIM, OUT_DIM).to(device=device)
    x2 = torch.rand(IN_DIM, OUT_DIM).to(device=device)
    x3 = torch.rand(IN_DIM, OUT_DIM).to(device=device)

    x1.requires_grad_()
    x2.requires_grad_()
    x3.requires_grad_()
    print(f"After init x1&2&3: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")

    ############
    # step1:
    ############
    print(f"\nStep1")

    # loss1
    print(f"Before Fwd x1: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")
    y1 = model(x1)
    print(f"After Fwd x1: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")

    print(f"Before loss1: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")
    loss1 = y1.sum()
    print(f"After loss1: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")

    # dx1
    backward_b(loss1, x1, model)

    # dw1
    backward_w(loss1, model)

    deallocate_output_tensor(x1)
    deallocate_output_tensor(y1)
    # del x1
    # del y1
    print(f"After del x1&y1: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")

    # print(f"\n Step1:collect:{gc.collect()}")
    # print(f"object: {gc.get_objects()}")
    # print(f"garbage: {gc.garbage}")

    ############
    # step2:
    ############
    print(f"\nStep2")

    # loss2
    y2 = model(x2)
    loss2 = y2.sum()

    # dx2
    backward_b(loss2, x2, model)

    # dw2
    backward_w(loss2, model)
    deallocate_output_tensor(x2)
    deallocate_output_tensor(y2)
    # del x2
    # del y2
    print(f"After del x2&y2: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")

    print(f"\n Step2:collect:{gc.collect()}")
    # print(f"object: {gc.get_objects()}")
    print(f"garbage: {gc.garbage}")

    ############
    # step3:
    ############

    print(f"\nStep3")

    # loss3
    y3 = model(x3)
    loss3 = y3.sum()

    # dx2
    backward_b(loss3, x3, model)

    # dw2
    backward_w(loss3, model)

    deallocate_output_tensor(x3)
    deallocate_output_tensor(y3)
    # del x3
    # del y3

    print(f"After del x3&y3: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")

    print(f"\n Step3:collect:{gc.collect()}")
    # print(f"object: {gc.get_objects()}")
    print(f"garbage: {gc.garbage}")


# del activation
def activation_dx_dw():
    device = "cuda:0"
    # model = nn.Linear(IN_DIM, OUT_DIM, bias=None).to(device=device)
    print(f"Before init Model: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")
    model = MlpModel().to(device=device)
    x1 = torch.rand(IN_DIM, OUT_DIM).to(device=device)
    x2 = torch.rand(IN_DIM, OUT_DIM).to(device=device)
    x3 = torch.rand(IN_DIM, OUT_DIM).to(device=device)

    x1.requires_grad_()
    x2.requires_grad_()
    x3.requires_grad_()
    print(f"After init Model, x1,x2,x3: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")

    activations = {}

    def register_hooks(module):
        def activation_hook(module, input, output):
            activations[f"{module.__class__.__name__}_{id(module)}"] = output.detach()

        def bwd_hook(module, grad_input, grad_output):
            del activations[f"{module.__class__.__name__}_{id(module)}"]

        module.register_forward_hook(activation_hook)
        module.register_backward_hook(bwd_hook)

    model.apply(register_hooks)

    ############
    # step1:
    ############
    print(f"\nStep1")

    # loss1
    loss1 = model(x1).sum()

    # dx1
    backward_b(loss1, x1, model)

    # dw1
    backward_w(loss1, model)

    del loss1, x1
    print(f"After del x1&y1: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")

    ############
    # step2:
    ############
    print(f"\nStep2")

    # loss2
    loss2 = model(x2).sum()

    # dx2
    backward_b(loss2, x2, model)

    # dw2
    backward_w(loss2, model)

    # deallocate_output_tensor(x2)
    # deallocate_output_tensor(loss2)
    del x2, loss2
    print(f"After del x2&y2: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")

    ############
    # step3:
    ############
    print(f"\nStep3")

    # loss3
    loss3 = model(x3).sum()

    # dx2
    backward_b(loss3, x3, model)

    # dw2
    backward_w(loss3, model)

    del x3, loss3

    print(f"After del x3&y3: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")


# text dx dw in model chunk
def model_chunk_dx_dw():
    device = "cuda:0"
    num_layers = 4
    print(f"Before init Model: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")
    model = MlpModel(in_dim=4096, out_dim=4096, num_layers=num_layers).to(device=device)
    x = torch.rand(4096, 4096).to(device=device)
    x.requires_grad_()

    model_chunk_0 = torch.nn.ModuleList()  # for layer 1 & 2
    model_chunk_1 = torch.nn.ModuleList()  # for layer 3 & 4

    for idx, sub_model in enumerate(model.layers):
        if idx < 2:
            model_chunk_0.append(sub_model).cuda()
        else:
            model_chunk_1.append(sub_model).cuda()

    print(f"After init Model & input: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")

    # Step1:chunk 0 fwd
    activation = dict()  # layer_id: activation
    out = x
    for i in range(len(model_chunk_0)):
        layer = model_chunk_0[i]
        activation[i] = layer(out)
    print(f"After chunk0 fwd: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")
    # Step2:chunk 1 fwd
    for i in range(len(model_chunk_1)):
        layer = model_chunk_0[i]
        activation[i + 2] = layer(out)
    print(f"After chunk1 fwd: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")

    # Step3:chunk 1 bwd b: dx=w*dy & bwd w:dw=x*dy
    # visit layer reversely
    for i in range(len(model_chunk_1) - 1, -1, -1):
        layer = model_chunk_1[i]
        global_layer_idx = i + 2
        prev_global_layer_idx = i + 1 if i + 1 > 0 else None
        i + 3 if i + 3 < 4 else None

        # bwd b
        if global_layer_idx == num_layers - 1:  # last layer in last chunk; calculate loss
            loss = activation[global_layer_idx].sum()
            x = activation[prev_global_layer_idx]
            backward_b(loss, x, layer)
        else:
            loss = activation[global_layer_idx].sum()
            x = activation[prev_global_layer_idx]
            backward_b(loss, x, layer)

        # bwd w
        backward_w(loss, layer)


def test_dx_dw_linear_benchmark():
    device = "cuda:0"
    model = nn.Linear(4096, 4096, bias=None).to(device=device)
    # print(f"model numel {get_model_numel(model)}") # 4GB
    x1 = torch.rand(4096, 4096).to(device=device)
    # x2 = torch.rand(4096, 4096).to(device=device)
    ref_model = deepcopy(model)
    ref_x1 = x1.clone()
    # ref_x2 = x1.clone()

    # first step
    x1.requires_grad_()
    # x2.requires_grad_()
    ref_x1.requires_grad_()
    # ref_x2.requires_grad_()

    # loss for dx_dw bwd
    loss1 = model(x1).sum()
    # loss2 = model(x2).sum()

    # loss for common bwd
    ref_model(ref_x1).sum()
    # ref_loss2 = ref_model(ref_x2).sum()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            f"/home/nvme-share/home/duanjunwen/ColossalAI/tests/test_pipeline/test_schedule"
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        # dx1
        torch.cuda.synchronize()
        bwd_b_start_time = time.time()
        backward_b(loss1, x1, model)
        bwd_b_end_time = time.time()
        print(f"loss_1 bwd B runtime {bwd_b_end_time - bwd_b_start_time}")

        for p in model.parameters():
            assert p.grad is None
        assert x1.grad is not None

        # dw1
        torch.cuda.synchronize()
        bwd_w_start_time = time.time()
        backward_w(loss1, model)
        bwd_w_end_time = time.time()
        print(f"loss_1 bwd W runtime {bwd_w_end_time - bwd_w_start_time}")
        for p in model.parameters():
            assert p.grad is not None

        # # common bwd 1
        # torch.cuda.synchronize()
        # comm_bwd_start_time = time.time()
        # ref_loss1.backward()
        # comm_bwd_end_time = time.time()
        # print(f"loss_1 comm bwd runtime {comm_bwd_end_time - comm_bwd_start_time}")


def test_dx_dw_attn_benchmark():
    device = "cuda:0"
    model = Attention(dim=4096).to(device=device)
    # print(f"model numel {get_model_numel(model)}") # 4GB
    x1 = torch.rand(1, 256, 4096).to(device=device)
    # x2 = torch.rand(1, 256, 4096).to(device=device)
    ref_model = deepcopy(model)
    ref_x1 = x1.clone()
    # ref_x2 = x1.clone()

    # first step
    x1.requires_grad_()
    # x2.requires_grad_()
    ref_x1.requires_grad_()
    # ref_x2.requires_grad_()

    # loss for dx_dw bwd
    loss1 = model(x1).sum()
    # loss2 = model(x2).sum()

    # loss for common bwd
    ref_model(ref_x1).sum()
    # ref_loss2 = ref_model(ref_x2).sum()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            f"/home/nvme-share/home/duanjunwen/ColossalAI/tests/test_pipeline/test_schedule"
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        # dx1
        torch.cuda.synchronize()
        bwd_b_start_time = time.time()
        backward_b(loss1, x1, model)
        bwd_b_end_time = time.time()
        print(f"loss_1 bwd B runtime {bwd_b_end_time - bwd_b_start_time}")

        for p in model.parameters():
            assert p.grad is None
        assert x1.grad is not None

        # dw1
        torch.cuda.synchronize()
        bwd_w_start_time = time.time()
        backward_w(loss1, model)
        bwd_w_end_time = time.time()
        print(f"loss_1 bwd W runtime {bwd_w_end_time - bwd_w_start_time}")
        for p in model.parameters():
            assert p.grad is not None

        # # common bwd 1
        # torch.cuda.synchronize()
        # comm_bwd_start_time = time.time()
        # ref_loss1.backward()
        # comm_bwd_end_time = time.time()
        # print(f"loss_1 comm bwd runtime {comm_bwd_end_time - comm_bwd_start_time}")


if __name__ == "__main__":
    # test_dx_dw_split()
    # test_double_dx_dw_split_nsync()
    # test_double_dx_dw_split_sync()
    # mem_dx_dw()
    # activation_dx_dw()
    # test_dx_dw_linear_benchmark()
    test_dx_dw_attn_benchmark()
