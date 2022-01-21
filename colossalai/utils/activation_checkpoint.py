#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
from torch.utils.checkpoint import check_backward_validity, detach_variable

from colossalai.context.random import get_states, get_current_mode, set_seed_states, set_mode, sync_states


class CheckpointFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, run_function, *args):
        check_backward_validity(args)
        ctx.run_function = run_function

        # preserve rng states
        ctx.fwd_cpu_rng_state = torch.get_rng_state()
        sync_states()
        ctx.fwd_seed_states = get_states(copy=True)
        ctx.fwd_current_mode = get_current_mode()

        if hasattr(torch, 'is_autocast_enabled'):
            ctx.had_autocast_in_fwd = torch.is_autocast_enabled()
        else:
            ctx.had_autocast_in_fwd = False

        # Save non-tensor inputs in ctx, keep a placeholder None for tensors
        # to be filled out during the backward.
        ctx.inputs = []
        ctx.tensor_indices = []
        tensor_inputs = []
        for i, arg in enumerate(args):
            if torch.is_tensor(arg):
                tensor_inputs.append(arg)
                ctx.tensor_indices.append(i)
                ctx.inputs.append(None)
            else:
                ctx.inputs.append(arg)

        ctx.save_for_backward(*tensor_inputs)

        with torch.no_grad():
            outputs = run_function(*args)
        return outputs

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad() or when an `inputs` parameter"
                " is passed to .backward(). Please use .backward() and do not pass its `inputs`"
                " argument.")
        # Copy the list to avoid modifying original list.
        inputs = list(ctx.inputs)
        tensor_indices = ctx.tensor_indices
        tensors = ctx.saved_tensors

        # store the current states
        bwd_cpu_rng_state = torch.get_rng_state()
        sync_states()
        bwd_seed_states = get_states(copy=True)
        bwd_current_mode = get_current_mode()

        # set the states to what it used to be
        torch.set_rng_state(ctx.fwd_cpu_rng_state)
        for parallel_mode, state in ctx.fwd_seed_states.items():
            set_seed_states(parallel_mode, state)
        set_mode(ctx.fwd_current_mode)

        # Fill in inputs with appropriate saved tensors.
        for i, idx in enumerate(tensor_indices):
            inputs[idx] = tensors[i]

        detached_inputs = detach_variable(tuple(inputs))
        if ctx.had_autocast_in_fwd:
            with torch.enable_grad(), torch.cuda.amp.autocast():
                outputs = ctx.run_function(*detached_inputs)
        else:
            with torch.enable_grad():
                outputs = ctx.run_function(*detached_inputs)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # recover the rng states
        torch.set_rng_state(bwd_cpu_rng_state)
        for parallel_mode, state in bwd_seed_states.items():
            set_seed_states(parallel_mode, state)
        set_mode(bwd_current_mode)

        # run backward() with only tensor that requires grad
        outputs_with_grad = []
        args_with_grad = []
        for i in range(len(outputs)):
            if torch.is_tensor(outputs[i]) and outputs[i].requires_grad:
                outputs_with_grad.append(outputs[i])
                args_with_grad.append(args[i])
        if len(outputs_with_grad) == 0:
            raise RuntimeError(
                "none of output has requires_grad=True,"
                " this checkpoint() is not necessary")
        torch.autograd.backward(outputs_with_grad, args_with_grad)
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else None
                      for inp in detached_inputs)

        return (None,) + grads


def checkpoint(function, *args):
    """Checkpoint the computation while preserve the rng states, modified from Pytorch torch.utils.checkpoint

    :param function: Describe the forward pass function. It should know how to handle the input tuples.
    :param args: Tuple containing the parameters of the function
    :return: Output of running function with provided args
    """
    return CheckpointFunction.apply(function, *args)
