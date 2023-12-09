import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F


class AllGatherLinearWithRingCommunication(torch.autograd.Function):
    """
    col-linear with hidden all_gather

    Y: [batch_size, seq_len / TP_size, hidden_size]
    A: [batch_size, hidden_size, w_len / TP_size]
                         |
                         | Ring-based LinearOverlap
                         v
    YA: [batch_size, seq_len, w_len / TP_size]
    """

    @staticmethod
    def forward(ctx, input_, weight, bias, process_group):
        # Input expected: (input_, bias) sharded on the row(sequence dim) and weight on the col
        ctx.save_for_backward(input_, weight, bias)
        ctx.use_bias = bias is not None
        ctx.process_group = process_group

        # if bias is not None:
        #    output = F.linear(input_, weight, bias)
        # else:
        #    output = F.linear(input_, weight)
        group_size = dist.get_world_size(process_group)
        cur_rank = dist.get_rank(process_group)

        input_shape = input_.shape
        weight_shape = weight.shape

        output_tensors = [torch.empty((input_shape[0], input_shape[1], weight_shape[0])) for _ in range(group_size)]
        # output_tensor = torch.empty((input_shape[0], input_shape[1] * group_size, weight_shape[0]), device=input_.device)

        # initialization of ring communication
        input_shape[1]
        recv_rank = cur_rank + 1 if cur_rank + 1 < group_size else 0
        send_rank = cur_rank - 1 if cur_rank > 0 else group_size - 1
        recv_tensor = input_.clone()
        send_tensor = input_.clone()
        input_tensor = input_.clone()
        # output_pt = output_tensor

        recv_op = dist.P2POp(dist.irecv, recv_tensor, recv_rank, group=process_group)
        send_op = dist.P2POp(dist.isend, send_tensor, send_rank, group=process_group)
        handles = dist.batch_isend_irecv([recv_op, send_op])
        # first round: special case, retrive from local tensor
        output_tensors[0] = F.linear(input_, weight)
        # output_pt[:][:local_seq_len][:] = F.linear(input_, weight)
        # output_pt = output_pt[:][local_seq_len:][:]
        for i in range(group_size - 2):
            handles[0].wait()
            handles[1].wait()

            tmp_tensor = input_tensor
            input_tensor = recv_tensor
            recv_tensor = tmp_tensor
            send_tensor = input_tensor.clone()

            recv_op = dist.P2POp(dist.irecv, recv_tensor, recv_rank, group=process_group)
            send_op = dist.P2POp(dist.isend, send_tensor, send_rank, group=process_group)
            handles = dist.batch_isend_irecv([recv_op, send_op])

            # actual computation
            output_tensors[i + 1] = F.linear(input_tensor, weight)
            # output_pt[:][:local_seq_len][:] = F.linear(input_, weight)
            # output_pt = output_pt[:][local_seq_len:][:]

        # final round: special case, no need to send/recv again
        handles[0].wait()
        # output_pt[:][:local_seq_len][:] = F.linear(recv_tensor, weight)
        output_tensors[group_size - 1] = F.linear(recv_tensor, weight)
        handles[1].wait()
        return torch.cat(output_tensors[group_size - cur_rank :] + output_tensors[: group_size - cur_rank], dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
        input, weight, bias = ctx.saved_tensors
        use_bias = ctx.use_bias

        # In order to be hooked into Gemini's '__torch_function__', adding a view operation to bias.
        if use_bias:
            bias.view(bias.shape)

        total_input = input
        grad_input = grad_output.matmul(weight)
        grad_output = grad_output.contiguous()
        # Convert the tensor shapes to 2D for execution compatibility
        if len(grad_output.shape) > 2:
            grad_output = grad_output.view(-1, grad_output.shape[-1])
            total_input = total_input.view(-1, total_input.shape[-1])

        if ctx.async_grad_allreduce:
            # Asynchronous all-reduce
            handle = dist.all_reduce(grad_input, group=ctx.process_group, async_op=True)
            # Delay the start of weight gradient computation shortly (3us) to have
            # all-reduce scheduled first and have GPU resources allocated
            _ = torch.empty(1, device=grad_output.device) + 1

        grad_weight = grad_output.t().matmul(total_input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if ctx.async_grad_allreduce:
            handle.wait()

        return grad_input, grad_weight, grad_bias, None, None, None


def main():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    scale_c = 4
    y = torch.randn(4, 5120 * scale_c, 1024 * scale_c, requires_grad=False).cuda()
    w = torch.randn(256 * scale_c, 1024 * scale_c, requires_grad=False).cuda()

    trial_time = 5

    ## warm up
    tensor_list = [torch.zeros_like(y) for _ in range(4)]
    dist.all_gather(tensor_list, y)

    Y = torch.cat(tensor_list, dim=1)
    torch_out = F.linear(Y, w)

    ring_out = AllGatherLinearWithRingCommunication.apply(y, w, None, None)
    ##

    tic = time.perf_counter()
    for _ in range(trial_time):
        tensor_list = [torch.zeros_like(y) for _ in range(4)]
        dist.all_gather(tensor_list, y)

        Y = torch.cat(tensor_list, dim=1)
        torch_out = F.linear(Y, w)
        print(torch_out[0][0][0])
    toc = time.perf_counter()
    print(f"original function in {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    for _ in range(trial_time):
        ring_out = AllGatherLinearWithRingCommunication.apply(y, w, None, None)
        print(ring_out[0][0][0])
    toc = time.perf_counter()
    print(f"fused function in {toc - tic:0.4f} seconds")

    if not torch.allclose(torch_out, ring_out, atol=1e-3):
        raise RuntimeError("ring_overlap: failed!!")
    print("ring_overlap: pass.")


if __name__ == "__main__":
    main()
