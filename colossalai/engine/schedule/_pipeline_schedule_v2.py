#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Tuple, Iterable

from colossalai import engine
import colossalai.communication.p2p_v2 as comm
import torch.cuda
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.utils.cuda import get_current_device

from ._pipeline_schedule import PipelineSchedule


def pack_return_tensors(return_tensors):
    output, label = tuple(zip(*return_tensors))
    if isinstance(output[0], torch.Tensor):
        output = torch.cat(output, dim=0)
    elif isinstance(output[0], (list, tuple)):
        output = tuple(torch.cat(tensors, dim=0) for tensors in zip(*output))
    else:
        raise TypeError(f'Output of model must be tensor or list/tuple of tensors')
    if isinstance(label[0], torch.Tensor):
        label = torch.cat(label, dim=0)
    else:
        merged_label = {k: [] for k in label[0].keys()}
        for d in label:
            for k, v in d.items():
                merged_label[k].append(v)
        label = {k: torch.cat(v, dim=0) for k, v in merged_label.items()}
    return output, label


class PipelineScheduleV2(PipelineSchedule):
    """Derived class of PipelineSchedule, the only difference is that
       forward_backward_step is reconstructed with p2p_v2
    
    Args:
        num_microbatches (int): The number of microbatches.
        data_process_func (Callable, optional):
            The preprocessing function which receives a batch of data, and it will be executed in `load_batch`.
        tensor_shape (torch.Size, optional): Specified shape in pipeline communication.
        scatter_gather_tensors (bool, optional):
            If set to `True`, communication will be reduced over pipeline when using 1D tensor parallelization.
    
    Example:
    
        # this shows an example of customized data_process_func
        def data_process_func(stage_output, dataloader_output):
            output1, output2 = stage_output
            item1, item2, item3 = dataloader_output

            # assume item2 is not needed
            data = (output1, output2, item1)
            label = item3
            return data, label

    """

    def forward_backward_step(self,
                              engine: engine.Engine,
                              data_iter: Iterable,
                              forward_only=False,
                              return_loss=True,
                              return_output_label=True) -> Tuple[torch.Tensor]:
        """Runs non-interleaved 1F1B schedule, with communication between pipeline stages.
        Returns a tuple with losses if the last stage, an empty tuple otherwise.

        Args:
            engine (colossalai.engine.Engine): Colossalai engine for training and inference.
            data_iter (Iterable): Dataloader as the form of an iterator, obtained by calling iter(dataloader).
            forward_only (bool, optional):
                Whether run forward step only. Default is false. If true, no backward will be run.
            return_loss (bool, optional): Whether returns the loss value. Default is true.
            return_output_label (bool, optional): If False, the output and label won't be returned.

        Returns:
            Tuple[:class:`torch.Tensor`]: A tuple of (output, label, loss), loss and label could be None.
        """

        assert forward_only or return_loss, \
            'The argument \'return_loss\' has to be True when \'forward_only\' is False, but got False.'
        self.load_batch(data_iter)

        # num_warmup_microbatches is the step when not all the processers are working
        num_warmup_microbatches = \
            (gpc.get_world_size(ParallelMode.PIPELINE)
             - gpc.get_local_rank(ParallelMode.PIPELINE) - 1)
        num_warmup_microbatches = min(num_warmup_microbatches, self.num_microbatches)
        num_microbatches_remaining = self.num_microbatches - num_warmup_microbatches

        # Input, output tensors only need to be saved when doing backward passes
        input_objs = None
        output_objs = None
        # local_rank = gpc.get_local_rank(ParallelMode.PIPELINE)

        if not forward_only:
            input_objs = []
            output_objs = []
        return_tensors = []
        if return_loss and gpc.is_pipeline_last_stage(ignore_virtual=True):
            accum_loss = torch.zeros(1, device=get_current_device())
        else:
            accum_loss = None

        # Run warmup forward passes.
        for i in range(num_warmup_microbatches):
            input_obj = comm.recv_forward()

            output_obj = self._forward_step(engine,
                                            input_obj,
                                            return_tensors,
                                            return_output_label=return_output_label,
                                            accum_loss=accum_loss)

            comm.send_forward(output_obj)

            if not forward_only:
                input_objs.append(input_obj)
                output_objs.append(output_obj)

        # Before running 1F1B, need to receive first forward tensor.
        # If all microbatches are run in warmup / cooldown phase, then no need to
        # receive this tensor here.
        if num_microbatches_remaining > 0:
            input_obj = comm.recv_forward()

        # Run 1F1B in steady state.
        for i in range(num_microbatches_remaining):
            last_iteration = (i == (num_microbatches_remaining - 1))

            output_obj = self._forward_step(engine,
                                            input_obj,
                                            return_tensors,
                                            return_output_label=return_output_label,
                                            accum_loss=accum_loss)
            if forward_only:
                comm.send_forward(output_obj)

                if not last_iteration:
                    input_obj = comm.recv_forward()

            else:
                # TODO adjust here
                comm.send_forward(output_obj)
                output_obj_grad = comm.recv_backward()

                # Add input_obj and output_obj to end of list.
                input_objs.append(input_obj)
                output_objs.append(output_obj)

                # Pop output_obj and output_obj from the start of the list for
                # the backward pass.
                input_obj = input_objs.pop(0)
                output_obj = output_objs.pop(0)

                input_obj_grad = self._backward_step(engine, input_obj, output_obj, output_obj_grad)

                if last_iteration:
                    input_obj = None
                    comm.send_backward(input_obj_grad)
                else:
                    input_obj = comm.recv_forward()
                    comm.send_backward(input_obj_grad)

        # Run cooldown backward passes.
        if not forward_only:
            for i in range(num_warmup_microbatches):
                input_obj = input_objs.pop(0)
                output_obj = output_objs.pop(0)

                output_obj_grad = comm.recv_backward()
                input_obj_grad = self._backward_step(engine, input_obj, output_obj, output_obj_grad)
                comm.send_backward(input_obj_grad)

        if len(return_tensors) > 0:
            output, label = pack_return_tensors(return_tensors)
            return output, label, accum_loss
        else:
            return None, None, accum_loss
