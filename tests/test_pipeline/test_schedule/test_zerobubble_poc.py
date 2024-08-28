import gc
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.testing import assert_close

import colossalai
from colossalai.cluster import ProcessGroupMesh
from colossalai.pipeline.p2p import PipelineP2PCommunication
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.testing import rerun_if_address_is_in_use, spawn

# info of model
IN_DIM = 8192
OUT_DIM = 8192
NUM_LAYER = 3


# A simple MLP
class MlpModel(nn.Module):
    def __init__(self, in_dim=IN_DIM, out_dim=OUT_DIM, num_layers=NUM_LAYER):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=None) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# Step1: dx = w*dy
def backward_b(loss, x, model):
    print(f"Before bwd b: {torch.cuda.memory_allocated()/1024**3 :.3f} GB")
    torch.autograd.backward(loss, inputs=x, retain_graph=True)
    print(f"After bwd b: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")


# Step1: dx = w*dy; for layer not last
def backward_b_not_last(tensors, grad, x, model):
    print(f"Before bwd b: {torch.cuda.memory_allocated()/1024**3 :.3f} GB")
    torch.autograd.backward(tensors=tensors, grad_tensors=grad, inputs=x, retain_graph=True)
    print(f"After bwd b: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")


def backward_w(loss, model):
    print(f"Before bwd w: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")
    torch.autograd.backward(loss, inputs=list(model.parameters()))
    print(f"After bwd w: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")


# Step2: dummy dw = x*dy
def backward_w_not_last(tensors, grad, model):
    print(f"Before bwd w: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")
    torch.autograd.backward(tensors=tensors, grad_tensors=grad, inputs=list(model.parameters()))
    print(f"After bwd w: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")


# In this poc, we check feasibility of spliting dx and dw in bwd propagation
def test_dx_dw_split():
    device = "cuda:0"
    model = nn.Linear(8, 8, bias=None).to(device=device)
    print(f"model numel {get_model_numel(model)}")  # 4GB
    x = torch.rand(8, 8).to(device=device)
    ref_model = deepcopy(model)
    ref_x = x.clone()

    # first step
    x.requires_grad_()
    loss = model(x).sum()
    backward_b(loss, x, model)
    for p in model.parameters():
        assert p.grad is None
    assert x.grad is not None
    backward_w(loss, model)
    for p in model.parameters():
        assert p.grad is not None

    # # second step
    # loss = model(x).sum()
    # backward_b(loss, x, model)
    # backward_w(loss, model)

    ref_x.requires_grad_()
    ref_loss = ref_model(ref_x).sum()
    ref_loss.backward()

    assert torch.equal(x.grad, ref_x.grad)
    for p1, p2 in zip(model.parameters(), ref_model.parameters()):
        assert torch.equal(p1.grad, p2.grad)


# In this poc, we check nsync of spliting dx and dw in bwd propagation in following order:
# fwd1 --> fwd2 --> dx1 --> dx2 --> dw1 --> dw2
def test_double_dx_dw_split_nsync():
    device = "cuda:0"
    model = nn.Linear(8, 8, bias=None).to(device=device)
    # print(f"model numel {get_model_numel(model)}") # 4GB
    x1 = torch.rand(8, 8).to(device=device)
    x2 = torch.rand(8, 8).to(device=device)
    ref_model = deepcopy(model)
    ref_x1 = x1.clone()
    ref_x2 = x2.clone()

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
    backward_b(loss1, x1, model)
    for p in model.parameters():
        assert p.grad is None
    assert x1.grad is not None

    # dx2
    backward_b(loss2, x2, model)

    # dw1
    backward_w(loss1, model)
    for p in model.parameters():
        assert p.grad is not None

    # common bwd 1
    ref_loss1.backward()

    # assert dx1 & dw1 == bwd 1
    assert_close(x1.grad, ref_x1.grad)
    for p1, p2 in zip(model.parameters(), ref_model.parameters()):
        assert_close(p1, p2)
        assert_close(p1.grad, p2.grad)

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


# In this poc, we check sync of spliting dx and dw in bwd propagation in following order:
# fwd1 --> fwd2 --> dx1 --> dw1 --> dx2 --> dw2
def test_double_dx_dw_split_sync():
    device = "cuda:0"
    model = nn.Linear(8, 8, bias=None).to(device=device)
    x1 = torch.rand(8, 8).to(device=device)
    x2 = torch.rand(8, 8).to(device=device)

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
    print(f"Step1\n")

    # loss1
    loss1 = model(x1).sum()

    # ref_loss1
    ref_loss1 = ref_model(ref_x1).sum()

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
    ref_loss1.backward()

    # assert dx1 & dw1 == bwd 1
    assert_close(x1.grad, ref_x1.grad)
    for p1, p2 in zip(model.parameters(), ref_model.parameters()):
        assert_close(p1, p2)
        assert_close(p1.grad, p2.grad)

    ############
    # step2:
    ############
    print(f"Step2\n")

    # loss2
    loss2 = model(x2).sum()

    # ref_loss2
    ref_loss2 = ref_model(ref_x2).sum()

    for p1, p2 in zip(model.parameters(), ref_model.parameters()):
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
        assert_close(p1, p2)
        assert_close(p1.grad, p2.grad)


# In this poc, we check if a memory leak has occurred after del input & loss(with graph)
def mem_dx_dw():
    device = "cuda:0"
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
    loss1 = model(x1).sum()
    print(f"After Fwd x1: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")

    print(f"Before loss1: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")
    print(f"After loss1: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")

    # dx1
    backward_b(loss1, x1, model)

    # dw1
    backward_w(loss1, model)

    del loss1, x1
    # del x1
    # del y1
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

    del x2, loss2
    # del x2
    # del y2
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

    # del x3
    # del y3
    del x3, loss3

    print(f"After del x3&y3: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")

    param_ids = [id(p) for p in model.parameters()]
    for obj in gc.get_objects():
        if torch.is_tensor(obj) and id(obj) not in param_ids:
            print(obj)


# In this poc, we check if a memory leak has occurred after del input & loss(with graph) & activation
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

    ############
    # step1:
    ############
    print(f"\nStep1")

    # loss1
    output1 = model(x1)
    loss1 = output1.sum()

    # dx1
    backward_b(loss1, x1, model)

    # dw1
    backward_w(loss1, model)

    # del loss1, x1
    del loss1, x1, output1
    print(f"After del : {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")

    ############
    # step2:
    ############
    print(f"\nStep2")

    # loss2
    output2 = model(x2)
    loss2 = output2.sum()

    # dx2
    backward_b(loss2, x2, model)

    # dw2
    backward_w(loss2, model)

    # del x2, loss2
    del x2, loss2, output2
    print(f"After del : {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")

    ############
    # step3:
    ############
    print(f"\nStep3")

    # loss3
    output3 = model(x3)
    loss3 = output3.sum()

    # dx2
    backward_b(loss3, x3, model)

    # dw2
    backward_w(loss3, model)

    # del x3, loss3
    del x3, loss3, output3

    print(f"After del : {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")


# In this poc, we apply model chunk instead of layer
def model_chunk_dx_dw():
    device = "cuda:0"
    num_layers = 4
    print(f"Before init Model: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")
    model = MlpModel(in_dim=4096, out_dim=4096, num_layers=num_layers).to(device=device)
    input = torch.rand(4096, 4096, requires_grad=True).to(device=device)

    input_base = input.clone()

    model_base = deepcopy(model)

    ##########################
    # Fwd bwd for dx dw
    ##########################

    model_chunk_0 = torch.nn.Sequential()  # for layer 1 & 2
    model_chunk_1 = torch.nn.Sequential()  # for layer 3 & 4

    for idx, sub_model in enumerate(model.layers):
        if idx < 2:
            model_chunk_0.append(sub_model)
        else:
            model_chunk_1.append(sub_model)

    print(f"After init Model & input: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")

    ##########################
    # Step1:chunk 0 fwd
    ##########################
    output1 = model_chunk_0(input)

    # detach output1; then output1 for chunk 0, output1_dt for chunk 1;
    output1_dt = output1.detach()
    output1_dt.requires_grad_()
    print(f"After chunk0 fwd (include detach output1): {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")

    ##########################
    # Step2:chunk 1 fwd
    ##########################
    output2 = model_chunk_1(output1_dt)

    print(f"After chunk1 fwd: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")

    ##########################
    # Step3:chunk 1 bwd b: dx=w*dy & bwd w:dw=x*dy
    ##########################
    loss = output2.mean()
    backward_b(loss, output1_dt, model_chunk_1)
    backward_w(loss, model_chunk_1)

    print(f"After chunk1 bwd b & w: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")

    ##########################
    # Step4:chunk 0 bwd b: dx=w*dy & bwd w:dw=x*dy
    ##########################
    # dx = w*dy
    backward_b_not_last(tensors=output1, grad=output1_dt.grad, x=input, model=model_chunk_0)
    backward_w_not_last(tensors=output1, grad=output1_dt.grad, model=model_chunk_0)

    print(f"After chunk0 bwd b & w: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")

    ##########################
    # Fwd bwd for base
    ##########################

    # fwd & bwd
    output_base = model_base(input_base)

    loss_base = output_base.mean()

    loss_base.backward()
    print(f"After base fwd & bwd: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")

    ##########################
    # Assert param
    ##########################

    assert_close(output2, output_base)
    assert_close(output2.grad, output_base.grad)

    for p1, p2 in zip(model.parameters(), model_base.parameters()):
        assert_close(p1, p2)
        assert_close(p1.grad, p2.grad)

    del output1, output1_dt, output2, loss, loss_base, output_base
    print(f"After del: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")


# In this poc, we apply model chunk  and a pp group for communication
def model_chunk_dx_dw_communication(
    rank: int,
    world_size: int,
    port: int,
):
    # init dist
    colossalai.launch(rank=rank, world_size=world_size, port=port, host="localhost")
    pg_mesh = ProcessGroupMesh(world_size)
    stage_manager = PipelineStageManager(pg_mesh, pipeline_axis=0, enable_interleave=True, num_model_chunks=2)
    rank = dist.get_rank()
    comm = PipelineP2PCommunication(stage_manager, overlap_p2p=False)

    print(f"{stage_manager.get_rank()}")

    # init model and input
    num_layers = 4
    print(f"Before init Model: {torch.cuda.memory_allocated()/1024**3 :.3f} GB on device {stage_manager.get_rank()};")
    model = MlpModel(in_dim=4096, out_dim=4096, num_layers=num_layers).to(rank)
    input = torch.rand(4096, 4096, requires_grad=True).to(rank)

    input_base = input.clone()
    model_base = deepcopy(model)

    if rank == 0:
        model_chunk_0 = torch.nn.Sequential().to(rank)  # for layer 1 & 2 on rank0
        for idx, sub_model in enumerate(model.layers):
            if idx < 2:
                model_chunk_0.append(sub_model)
    else:
        model_chunk_1 = torch.nn.Sequential().to(rank)  # for layer 3 & 4 on rank1
        for idx, sub_model in enumerate(model.layers):
            if idx >= 2:
                model_chunk_1.append(sub_model)

    print(
        f"After init Model & input: {torch.cuda.memory_allocated()/1024**3 :.3f} GB on device {stage_manager.get_rank()};"
    )

    ##########################
    # Step1:chunk 0 fwd
    ##########################
    if rank == 0:
        output1 = model_chunk_0(input)
        print(
            f"After chunk0 fwd (include detach output1): {torch.cuda.memory_allocated()/1024**3 :.3f} GB on device {stage_manager.get_rank()};"
        )
        # send y(output1_dt) to next stage
        comm.send_forward(output1, stage_manager.get_next_rank())

    ##########################
    # Step2:chunk 1 fwd
    ##########################
    if rank == 1:
        # recv y(output1_dt) from prev stage
        output1_dt_rank1, wait_handles = comm.recv_forward(stage_manager.get_prev_rank())
        output1_dt_rank1.requires_grad_()
        output2 = model_chunk_1(output1_dt_rank1)

        print(
            f"After chunk1 fwd: {torch.cuda.memory_allocated()/1024**3 :.3f} GB on device {stage_manager.get_rank()};"
        )

    ##########################
    # Step3:chunk 1 on device_1 bwd b: dx=w*dy & bwd w:dw=x*dy
    ##########################
    if rank == 1:
        loss = output2.mean()
        backward_b(loss, output1_dt_rank1, model_chunk_1)
        backward_w(loss, model_chunk_1)

        print(f"After chunk1 bwd b & w: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")
        # send bwd output1_dt_rank1 from rank1 to rank 0
        comm.send_backward(output1_dt_rank1.grad, stage_manager.get_prev_rank())
    ##########################
    # Step4:chunk 0 on device_0 bwd b: dx=w*dy & bwd w:dw=x*dy
    ##########################

    if rank == 0:
        # recv bwd output1_dt_rank1 from rank1 to rank 0
        output1_dt_rank0_grad, _ = comm.recv_backward(stage_manager.get_next_rank())

        backward_b_not_last(tensors=output1, grad=output1_dt_rank0_grad, x=input, model=model_chunk_0)
        backward_w_not_last(tensors=output1, grad=output1_dt_rank0_grad, model=model_chunk_0)

        print(f"After chunk0 bwd b & w: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")

    ##########################
    # Fwd bwd for base
    ##########################
    # fwd & bwd
    output_base = model_base(input_base)
    loss_base = output_base.mean()
    loss_base.backward()
    print(f"After base fwd & bwd: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")

    ##########################
    # Assert param
    ##########################
    # assert output
    if rank == 1:
        assert_close(output2, output_base)
        assert_close(output2.grad, output_base.grad)

    # assert model param & grad
    if rank == 0:
        count = 0
        for (chunk_name, chunk_param), (base_name, base_param) in zip(
            model_chunk_0.named_parameters(), model_base.named_parameters()
        ):
            if count < 2:
                assert_close(chunk_param, base_param)
                assert_close(chunk_param.grad, base_param.grad)
            count += 1
    if rank == 1:
        count = 0
        for (chunk_name, chunk_param), (base_name, base_param) in zip(
            model_chunk_1.named_parameters(), model_base.named_parameters()
        ):
            if count >= 2:
                assert_close(chunk_param, base_param)
                assert_close(chunk_param.grad, base_param.grad)
            count += 1
    # clean memory
    if rank == 0:
        del output1, output1_dt_rank0_grad
    if rank == 1:
        del output2, loss, output1_dt_rank1
    del loss_base, output_base
    print(f"After del: {torch.cuda.memory_allocated()/1024**3 :.3f} GB on device {stage_manager.get_rank()};")


# fwd schedule
def schedule_f(
    stage_manager: PipelineStageManager,
    comm: PipelineP2PCommunication,
    input: torch.Tensor,
    model_chunk: torch.nn.ModuleList,
    model_chunk_id: int,
):
    # chunk_id == 0
    if model_chunk_id == 0:
        # recv fwd from prev
        if stage_manager.is_first_stage(ignore_chunk=True):
            input = input  # get local input
        else:
            prev_rank = stage_manager.get_prev_rank()
            input, wait_handles = comm.recv_forward(prev_rank)

        # fwd step
        output = model_chunk[model_chunk_id](input)

        # send fwd to next
        if stage_manager.is_last_stage(ignore_chunk=True):
            return input, output, None  # return local output
        else:
            next_rank = stage_manager.get_next_rank()
            comm.send_forward(output, next_rank)

    # chunk_id == 1
    if model_chunk_id == 1:
        # recv fwd from next
        if stage_manager.is_last_stage(ignore_chunk=True):
            input = input  # get local input
        else:
            next_rank = stage_manager.get_next_rank()
            input, wait_handles = comm.recv_forward(next_rank)

        # fwd step
        output = model_chunk[model_chunk_id](input)

        # send fwd to prev
        if stage_manager.is_first_stage(ignore_chunk=True):
            loss = output.mean()
            return input, output, loss  # return local output
        else:
            prev_rank = stage_manager.get_prev_rank()
            comm.send_forward(output, prev_rank)
    return input, output, None


# bwd b schedule
def schedule_b(
    stage_manager: PipelineStageManager,
    comm: PipelineP2PCommunication,
    input: torch.Tensor,  # x
    output: torch.Tensor,  # y
    output_grad: torch.Tensor,  # dy
    model_chunk: torch.nn.ModuleList,
    model_chunk_id: int,
):
    # chunk_id == 0
    if model_chunk_id == 0:

        # recv bwd from next
        if stage_manager.is_last_stage(ignore_chunk=True):
            output_grad = output_grad  # get dy from local
        else:
            next_rank = stage_manager.get_next_rank()
            output_grad, _ = comm.recv_backward(next_rank)

        # bwd step
        backward_b_not_last(tensors=output, grad=output_grad, x=input, model=model_chunk[model_chunk_id])
        backward_w_not_last(tensors=output, grad=output_grad, model=model_chunk[model_chunk_id])

        # send bwd to prev
        if stage_manager.is_first_stage(ignore_chunk=True):
            return input.grad
        else:
            prev_rank = stage_manager.get_prev_rank()
            comm.send_backward(input.grad, prev_rank)

    # chunk_id == 1
    if model_chunk_id == 1:
        # recv bwd from prev
        if stage_manager.is_first_stage(ignore_chunk=True):
            output_grad = output_grad
        else:
            prev_rank = stage_manager.get_prev_rank()
            output_grad, _ = comm.recv_backward(next_rank=prev_rank)

        # bwd step
        if stage_manager.is_first_stage(ignore_chunk=True):
            backward_b(loss=output_grad, x=input, model=model_chunk[model_chunk_id])
            backward_w(loss=output_grad, model=model_chunk[model_chunk_id])
        else:
            # commom bwd step
            backward_b_not_last(tensors=output, grad=output_grad, x=input, model=model_chunk[model_chunk_id])
            backward_w_not_last(tensors=output, grad=output_grad, model=model_chunk[model_chunk_id])

        # send bwd to next
        if stage_manager.is_last_stage(ignore_chunk=True):
            return input.grad
        else:
            next_rank = stage_manager.get_next_rank()
            comm.send_backward(input.grad, next_rank)

    return input.grad


# bwd w schedule (dw already splite in schedule b)
def schedule_w():
    pass


# In this poc, we apply a scheduling method for each rank: schedule_f --> schedule_b --> schedule_w
def model_chunk_dx_dw_comm_interleaved(
    rank: int,
    world_size: int,
    port: int,
):
    # init dist
    colossalai.launch(rank=rank, world_size=world_size, port=port, host="localhost")
    pg_mesh = ProcessGroupMesh(world_size)
    stage_manager = PipelineStageManager(pg_mesh, pipeline_axis=0, enable_interleave=True, num_model_chunks=world_size)
    rank = dist.get_rank()
    comm = PipelineP2PCommunication(stage_manager, overlap_p2p=False)

    # init model and input
    num_layers = 8
    in_dim = out_dim = 2048
    print(f"Before init Model: {torch.cuda.memory_allocated()/1024**3 :.3f} GB on device {stage_manager.get_rank()};")
    model = MlpModel(in_dim=in_dim, out_dim=out_dim, num_layers=num_layers).to(rank)
    input0 = torch.rand(in_dim, out_dim, requires_grad=True).to(rank)

    input_base = input0.clone()
    model_base = deepcopy(model)

    if rank == 0:
        # layer 0 & 7 to chunk 0 on rank0
        chunk_0 = torch.nn.ModuleList().to(rank)
        for idx, sub_model in enumerate(model.layers):
            if idx == 0 or idx == 7:
                chunk_0.append(sub_model)
    elif rank == 1:
        # layer 1 & 6 to chunk 1 on rank1
        chunk_1 = torch.nn.ModuleList().to(rank)
        for idx, sub_model in enumerate(model.layers):
            if idx == 1 or idx == 6:
                chunk_1.append(sub_model)
    elif rank == 2:
        # layer 2 & 5 to chunk 2 on rank2
        chunk_2 = torch.nn.ModuleList().to(rank)
        for idx, sub_model in enumerate(model.layers):
            if idx == 2 or idx == 5:
                chunk_2.append(sub_model)
    else:
        # layer 3 & 4 to chunk 3 on rank3
        chunk_3 = torch.nn.Sequential().to(rank)
        for idx, sub_model in enumerate(model.layers):
            if idx == 3 or idx == 4:
                chunk_3.append(sub_model)

    print(
        f"After init Model & input: {torch.cuda.memory_allocated()/1024**3 :.3f} GB on device {stage_manager.get_rank()};"
    )
    # buffer use to save input and output

    ##########################
    # Step1: fwd
    ##########################
    ######
    # fwd 1->4
    ######
    # chunk 0 id 0 (layer 0) fwd
    if rank == 0:
        chunk_id = 0
        input0, output0, _ = schedule_f(
            stage_manager=stage_manager,
            comm=comm,
            input=input0,
            model_chunk=chunk_0,
            model_chunk_id=chunk_id,
        )
        print(
            f"chunk 0 id 0 (layer 0)fwd: {torch.cuda.memory_allocated()/1024**3 :.3f} GB on device {stage_manager.get_rank()};"
        )

    # chunk 1 id 0 (layer 1)  fwd
    if rank == 1:
        chunk_id = 0
        input1, output1, _ = schedule_f(
            stage_manager=stage_manager,
            comm=comm,
            input=None,
            model_chunk=chunk_1,
            model_chunk_id=chunk_id,
        )
        print(
            f"chunk 1 id 0 (layer 1)fwd: {torch.cuda.memory_allocated()/1024**3 :.3f} GB on device {stage_manager.get_rank()};"
        )

    # chunk 2 id 0 (layer 2)  fwd
    if rank == 2:
        chunk_id = 0
        input2, output2, _ = schedule_f(
            stage_manager=stage_manager,
            comm=comm,
            input=None,
            model_chunk=chunk_2,
            model_chunk_id=chunk_id,
        )
        print(
            f"chunk 2 id 0 (layer 2)fwd: {torch.cuda.memory_allocated()/1024**3 :.3f} GB on device {stage_manager.get_rank()};"
        )

    # chunk 3 id 0 (layer 3)  fwd
    if rank == 3:
        chunk_id = 0
        input3, output3, _ = schedule_f(
            stage_manager=stage_manager,
            comm=comm,
            input=None,
            model_chunk=chunk_3,
            model_chunk_id=chunk_id,
        )
        print(
            f"chunk 3 id 0 (layer 3)fwd: {torch.cuda.memory_allocated()/1024**3 :.3f} GB on device {stage_manager.get_rank()};"
        )

        ######
        # fwd 4->1
        ######

    if rank == 3:
        chunk_id = 1
        input4, output4, _ = schedule_f(
            stage_manager=stage_manager,
            comm=comm,
            input=output3,
            model_chunk=chunk_3,
            model_chunk_id=chunk_id,
        )
        print(
            f"chunk 3 id 1 (layer 4)fwd: {torch.cuda.memory_allocated()/1024**3 :.3f} GB on device {stage_manager.get_rank()};"
        )

    if rank == 2:
        chunk_id = 1
        input5, output5, _ = schedule_f(
            stage_manager=stage_manager,
            comm=comm,
            input=None,
            model_chunk=chunk_2,
            model_chunk_id=chunk_id,
        )
        print(
            f"chunk 2 id 1 (layer 5)fwd: {torch.cuda.memory_allocated()/1024**3 :.3f} GB on device {stage_manager.get_rank()};"
        )

    if rank == 1:
        chunk_id = 1
        input6, output6, _ = schedule_f(
            stage_manager=stage_manager,
            comm=comm,
            input=None,
            model_chunk=chunk_1,
            model_chunk_id=chunk_id,
        )
        print(
            f"chunk 1 id 1 (layer 6)fwd: {torch.cuda.memory_allocated()/1024**3 :.3f} GB on device {stage_manager.get_rank()};"
        )

    if rank == 0:
        chunk_id = 1
        input7, output7, loss = schedule_f(
            stage_manager=stage_manager,
            comm=comm,
            input=None,
            model_chunk=chunk_0,
            model_chunk_id=chunk_id,
        )
        # print(f"fwd output {output7}")
        print(
            f"chunk 0 id 1 (layer 7)fwd: {torch.cuda.memory_allocated()/1024**3 :.3f} GB on device {stage_manager.get_rank()};"
        )

    ##########################
    # Step2: bwd
    ##########################
    ######
    # bwd rank 4->1
    ######
    # chunk 0 id 1 (layer 7) bwd
    if rank == 0:
        chunk_id = 1
        input_grad7 = schedule_b(
            stage_manager=stage_manager,
            comm=comm,
            input=input7,  # x
            output=output7,  # y
            output_grad=loss,  # dy
            model_chunk=chunk_0,
            model_chunk_id=chunk_id,
        )

    # # chunk 1 id 1 (layer 6) bwd
    if rank == 1:
        chunk_id = 1
        input_grad6 = schedule_b(
            stage_manager=stage_manager,
            comm=comm,
            input=input6,  # x
            output=output6,  # y
            output_grad=None,  # dy
            model_chunk=chunk_1,
            model_chunk_id=chunk_id,
        )

    # chunk 2 id 1 (layer 5) bwd
    if rank == 2:
        chunk_id = 1
        input_grad5 = schedule_b(
            stage_manager=stage_manager,
            comm=comm,
            input=input5,  # x
            output=output5,  # y
            output_grad=None,  # dy
            model_chunk=chunk_2,
            model_chunk_id=chunk_id,
        )

    # chunk 3 id 1 (layer 4) bwd
    if rank == 3:
        chunk_id = 1
        input_grad4 = schedule_b(
            stage_manager=stage_manager,
            comm=comm,
            input=input4,  # x
            output=output4,  # y
            output_grad=None,  # dy
            model_chunk=chunk_3,
            model_chunk_id=chunk_id,
        )

        ######
        # bwd rank 1->4
        ######

    # chunk 3 id 0 (layer 3) bwd
    if rank == 3:
        chunk_id = 0
        input_grad3 = schedule_b(
            stage_manager=stage_manager,
            comm=comm,
            input=input3,  # x
            output=output3,  # y
            output_grad=input_grad4,  # dy
            model_chunk=chunk_3,
            model_chunk_id=chunk_id,
        )

    # chunk 2 id 0 (layer 2) bwd
    if rank == 2:
        chunk_id = 0
        input_grad2 = schedule_b(
            stage_manager=stage_manager,
            comm=comm,
            input=input2,  # x
            output=output2,  # y
            output_grad=None,  # dy
            model_chunk=chunk_2,
            model_chunk_id=chunk_id,
        )

    # chunk 1 id 0 (layer 1) bwd
    if rank == 1:
        chunk_id = 0
        input_grad1 = schedule_b(
            stage_manager=stage_manager,
            comm=comm,
            input=input1,  # x
            output=output1,  # y
            output_grad=None,  # dy
            model_chunk=chunk_1,
            model_chunk_id=chunk_id,
        )

    # chunk 0 id 0 (layer 0) bwd
    if rank == 0:
        chunk_id = 0
        input_grad0 = schedule_b(
            stage_manager=stage_manager,
            comm=comm,
            input=input0,  # x
            output=output0,  # y
            output_grad=None,  # dy
            model_chunk=chunk_0,
            model_chunk_id=chunk_id,
        )

    ##########################
    # Fwd bwd for base
    ##########################
    # fwd & bwd
    output_base = model_base(input_base)
    loss_base = output_base.mean()
    loss_base.backward()
    print(f"After base fwd & bwd: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")

    ##########################
    # Assert close
    ##########################
    # assert output
    if rank == 0:
        assert_close(output7, output_base)

    # assert weight
    if rank == 0:
        # layer 0
        assert_close(chunk_0[0].weight, model_base.layers[0].weight)
        assert_close(chunk_0[0].weight.grad, model_base.layers[0].weight.grad)
        # layer 7
        assert_close(chunk_0[1].weight, model_base.layers[7].weight)
        assert_close(chunk_0[1].weight.grad, model_base.layers[7].weight.grad)
    if rank == 1:
        # layer 1
        assert_close(chunk_1[0].weight, model_base.layers[1].weight)
        assert_close(chunk_1[0].weight.grad, model_base.layers[1].weight.grad)
        # layer 6
        assert_close(chunk_1[1].weight, model_base.layers[6].weight)
        assert_close(chunk_1[1].weight.grad, model_base.layers[6].weight.grad)

    if rank == 2:
        # layer 2
        assert_close(chunk_2[0].weight, model_base.layers[2].weight)
        assert_close(chunk_2[0].weight.grad, model_base.layers[2].weight.grad)
        # layer 5
        assert_close(chunk_2[1].weight, model_base.layers[5].weight)
        assert_close(chunk_2[1].weight.grad, model_base.layers[5].weight.grad)

    if rank == 3:
        # layer 3
        assert_close(chunk_3[0].weight, model_base.layers[3].weight)
        assert_close(chunk_3[0].weight.grad, model_base.layers[3].weight.grad)
        # layer 4
        assert_close(chunk_3[1].weight, model_base.layers[4].weight)
        assert_close(chunk_3[1].weight.grad, model_base.layers[4].weight.grad)

    # clean memory
    if rank == 0:
        del input0, output0, input_grad0, input7, output7, input_grad7, loss
    if rank == 1:
        del input1, output1, input_grad1, input6, output6, input_grad6
    if rank == 2:
        del input2, output2, input_grad2, input5, output5, input_grad5
    if rank == 3:
        del input3, output3, input_grad3, input4, output4, input_grad4
    del loss_base, output_base

    print(f"After del: {torch.cuda.memory_allocated()/1024**3 :.3f} GB on device {stage_manager.get_rank()};")


@rerun_if_address_is_in_use()
def test_dx_dw_dist():
    spawn(
        model_chunk_dx_dw_comm_interleaved,
        nprocs=4,
    )


if __name__ == "__main__":
    test_dx_dw_dist()
