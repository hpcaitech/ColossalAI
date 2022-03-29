#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import inspect
from typing import Callable, List, Tuple, Union

import colossalai.communication as comm
import torch.cuda
from colossalai.amp.naive_amp import NaiveAMPModel
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.utils import switch_virtual_pipeline_parallel_rank
from colossalai.utils.cuda import get_current_device
from colossalai.zero.sharded_model import ShardedModelV2

from ._base_schedule import BaseSchedule


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


class PipelineSchedule(BaseSchedule):
    """A helper schedule class for pipeline parallelism running environment.
    It uses non-interleaved 1F1B strategy. Other properties are similar as
    :class:`NonPipelineSchedule`.

    Args:
        num_microbatches (int): The number of microbatches.
        batch_data_process_func (Callable, optional):
            The preprocessing function which receives a batch of data, and it will be executed in `load_batch`.
        tensor_shape (torch.Size, optional): Specified shape in pipeline communication.
        scatter_gather_tensors (bool, optional):
            If set to `True`, communication will be reduced over pipeline when using 1D tensor parallelization.
    """

    def __init__(self,
                 num_microbatches,
                 batch_data_process_func: Callable = None,
                 tensor_shape: Union[torch.Size, List[int], Tuple[int]] = None,
                 scatter_gather_tensors: bool = False):
        super().__init__(batch_data_process_func=batch_data_process_func)
        self.num_microbatches = num_microbatches
        self.dtype = torch.float
        self.tensor_shape = tensor_shape
        self.scatter_gather_tensors = False
        if gpc.is_initialized(ParallelMode.PARALLEL_1D) and gpc.get_world_size(ParallelMode.PARALLEL_1D) > 1:
            self.scatter_gather_tensors = scatter_gather_tensors
        self._logger = get_dist_logger()

    def load_batch(self, data_iter):
        # Pipeline schedule just puts data in memory
        self.batch_data, self.batch_label = super().load_batch(data_iter, to_gpu=False)
        self.microbatch_offset = 0
        if isinstance(self.batch_data, torch.Tensor):
            batch_size = self.batch_data.size(0)
        else:
            batch_size = next(iter(self.batch_data.values())).size(0)
        assert batch_size % self.num_microbatches == 0, \
            "Batch size should divided by the number of microbatches"
        self.microbatch_size = batch_size // self.num_microbatches

    def _get_data_slice(self, data, offset):
        if isinstance(data, torch.Tensor):
            return data[offset:offset + self.microbatch_size]
        elif isinstance(data, dict):
            return {k: v[offset:offset + self.microbatch_size] for k, v in data.items()}

    def load_micro_batch(self):
        data = self._get_data_slice(self.batch_data, self.microbatch_offset)
        label = self._get_data_slice(self.batch_label, self.microbatch_offset)
        self.microbatch_offset += self.microbatch_size
        return self._move_to_device(data), self._move_to_device(label)

    def pre_processing(self, engine):
        # TODO: remove this after testing new zero with pipeline parallelism
        model = engine.model
        if isinstance(model, (NaiveAMPModel, ShardedModelV2)):
            self.dtype = torch.half
            model = model.model
        sig = inspect.signature(model.forward)
        for p in sig.parameters.values():
            assert p.kind != inspect.Parameter.VAR_POSITIONAL, '*args is not supported'

    @staticmethod
    def _call_engine(model, input_tensor, batch_data):
        if isinstance(model, NaiveAMPModel):
            sig = inspect.signature(model.model.forward)
        elif isinstance(model, ShardedModelV2):
            sig = inspect.signature(model.module.forward)
        else:
            sig = inspect.signature(model.forward)
        if isinstance(batch_data, torch.Tensor):
            if input_tensor is None:
                return model(batch_data)
            elif len(sig.parameters) > 1:
                return model(input_tensor, batch_data)
            else:
                return model(input_tensor)
        else:
            filter_batch = True
            for p in sig.parameters.values():
                if p.kind == inspect.Parameter.VAR_KEYWORD:
                    filter_batch = False
            if filter_batch:
                batch_data = {k: v for k, v in batch_data.items() if k in sig.parameters}
            if input_tensor is None:
                return model(**batch_data)
            else:
                return model(input_tensor, **batch_data)

    def forward_step(self, engine, input_tensor, return_tensors, return_output_label=True, accum_loss=None):
        """Forward step for passed-in model. If it is the first stage, the input tensor 
        is obtained from data_iterator, otherwise the passed-in input_tensor is used.
        Returns output tensor. This is a helper function and can be ignored by users.

        Args:
            engine (colossalai.engine.Engine): Colossalai engine for training and inference.
            input_tensor (:class:`torch.Tensor`): Input tensor for this pipeline stage.
            return_tensors (List[:class:`torch.Tensor`]): A list of tensors to return.
            return_output_label (bool, optional): Whether returns output labels.
            accum_loss (optional): Where accumulated loss stores.
        Returns:
            :class:`torch.Tensor`: output or the loss value of the current pipeline stage.
        """
        data, label = self.load_micro_batch()
        output_tensor = self._call_engine(engine.model, input_tensor, data)

        if gpc.is_last_rank(ParallelMode.PIPELINE):
            if return_output_label:
                return_tensors.append((output_tensor, label))
            if accum_loss is not None:
                loss_reduced = self._call_engine_criterion(engine, output_tensor, label) / self.num_microbatches
                accum_loss.add_(loss_reduced.detach())
                return loss_reduced
            else:
                # forward only, it's useless since backward is not needed
                return output_tensor
        else:
            assert isinstance(
                output_tensor,
                torch.Tensor), 'Output of model using pipeline parallelism must be a tensor (except the last stage).'
            self._logger.debug(
                f'Global rank {gpc.get_global_rank()}, pipeline rank {gpc.get_local_rank(ParallelMode.PIPELINE)} forward output tensor {output_tensor.shape}, dtype {output_tensor.dtype}'
            )
            return output_tensor

    def backward_step(self, engine, input_tensor, output_tensor, output_tensor_grad):
        """Backward step through the passed-in output tensor. If it is the last stage, the 
        output_tensor_grad is None, otherwise it is the gradients with respect to stage's output tensor.
        Returns the gradients with respect to the input tensor (None if first stage).
        This is a helper function and can be ignored by users.

        Args:
            engine (colossalai.engine.Engine): Colossalai engine for training and inference.
            input_tensor (:class:`torch.Tensor`): input tensor for this pipeline stage.
            output_tensor (:class:`torch.Tensor`): output tensor for this pipeline stage.
            output_tensor_grad (:class:`torch.Tensor`): gradient of output tensor for this pipeline stage.

        Returns:
            :class:`torch.Tensor`: gradient of input tensor.
        """

        # Retain the grad on the input_tensor.
        if input_tensor is not None:
            input_tensor.retain_grad()

        # Backward pass.
        if output_tensor_grad is None:
            engine.backward(output_tensor)
        else:
            engine.backward_by_grad(output_tensor, output_tensor_grad)

        # Collect the grad of the input_tensor.
        input_tensor_grad = None
        if input_tensor is not None:
            input_tensor_grad = input_tensor.grad

        return input_tensor_grad

    def forward_backward_step(self, engine, data_iter, forward_only=False, return_loss=True, return_output_label=True):
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
        num_warmup_microbatches = \
            (gpc.get_world_size(ParallelMode.PIPELINE)
             - gpc.get_local_rank(ParallelMode.PIPELINE) - 1)
        num_warmup_microbatches = min(num_warmup_microbatches, self.num_microbatches)
        num_microbatches_remaining = self.num_microbatches - num_warmup_microbatches

        # Input, output tensors only need to be saved when doing backward passes
        input_tensors = None
        output_tensors = None
        if not forward_only:
            input_tensors = []
            output_tensors = []
        return_tensors = []
        if return_loss and gpc.is_pipeline_last_stage(ignore_virtual=True):
            accum_loss = torch.zeros(1, device=get_current_device())
        else:
            accum_loss = None
        # Used for tensor meta information communication
        ft_shape = self.tensor_shape
        bt_shape = None
        fs_checker = self.tensor_shape is None

        # Run warmup forward passes.
        for i in range(num_warmup_microbatches):
            if not gpc.is_first_rank(ParallelMode.PIPELINE):
                ft_shape = comm.recv_tensor_meta(ft_shape)
            input_tensor = comm.recv_forward(ft_shape,
                                             dtype=self.dtype,
                                             scatter_gather_tensors=self.scatter_gather_tensors)
            output_tensor = self.forward_step(engine,
                                              input_tensor,
                                              return_tensors,
                                              return_output_label=return_output_label,
                                              accum_loss=accum_loss)
            if not gpc.is_last_rank(ParallelMode.PIPELINE):
                bt_shape = output_tensor.shape
                fs_checker = comm.send_tensor_meta(output_tensor, fs_checker)
            comm.send_forward(output_tensor, scatter_gather_tensors=self.scatter_gather_tensors)

            if not forward_only:
                input_tensors.append(input_tensor)
                output_tensors.append(output_tensor)

        # Before running 1F1B, need to receive first forward tensor.
        # If all microbatches are run in warmup / cooldown phase, then no need to
        # receive this tensor here.
        if num_microbatches_remaining > 0:
            if not gpc.is_first_rank(ParallelMode.PIPELINE):
                ft_shape = comm.recv_tensor_meta(ft_shape)
            input_tensor = comm.recv_forward(ft_shape,
                                             dtype=self.dtype,
                                             scatter_gather_tensors=self.scatter_gather_tensors)

        # Run 1F1B in steady state.
        for i in range(num_microbatches_remaining):
            last_iteration = (i == (num_microbatches_remaining - 1))

            output_tensor = self.forward_step(engine,
                                              input_tensor,
                                              return_tensors,
                                              return_output_label=return_output_label,
                                              accum_loss=accum_loss)
            if forward_only:
                comm.send_forward(output_tensor, scatter_gather_tensors=self.scatter_gather_tensors)

                if not last_iteration:
                    input_tensor = comm.recv_forward(ft_shape,
                                                     dtype=self.dtype,
                                                     scatter_gather_tensors=self.scatter_gather_tensors)

            else:
                output_tensor_grad = comm.send_forward_recv_backward(output_tensor,
                                                                     bt_shape,
                                                                     dtype=self.dtype,
                                                                     scatter_gather_tensors=self.scatter_gather_tensors)

                # Add input_tensor and output_tensor to end of list.
                input_tensors.append(input_tensor)
                output_tensors.append(output_tensor)

                # Pop input_tensor and output_tensor from the start of the list for
                # the backward pass.
                input_tensor = input_tensors.pop(0)
                output_tensor = output_tensors.pop(0)

                input_tensor_grad = self.backward_step(engine, input_tensor, output_tensor, output_tensor_grad)

                if last_iteration:
                    input_tensor = None
                    comm.send_backward(input_tensor_grad, scatter_gather_tensors=self.scatter_gather_tensors)
                else:
                    input_tensor = comm.send_backward_recv_forward(input_tensor_grad,
                                                                   ft_shape,
                                                                   dtype=self.dtype,
                                                                   scatter_gather_tensors=self.scatter_gather_tensors)

        # Run cooldown backward passes.
        if not forward_only:
            for i in range(num_warmup_microbatches):
                input_tensor = input_tensors.pop(0)
                output_tensor = output_tensors.pop(0)

                output_tensor_grad = comm.recv_backward(bt_shape,
                                                        dtype=self.dtype,
                                                        scatter_gather_tensors=self.scatter_gather_tensors)

                input_tensor_grad = self.backward_step(engine, input_tensor, output_tensor, output_tensor_grad)

                comm.send_backward(input_tensor_grad, scatter_gather_tensors=self.scatter_gather_tensors)

        if len(return_tensors) > 0:
            output, label = pack_return_tensors(return_tensors)
            return output, label, accum_loss
        else:
            return None, None, accum_loss


class InterleavedPipelineSchedule(PipelineSchedule):

    def __init__(self,
                 num_microbatches,
                 num_model_chunks,
                 batch_data_process_func: Callable = None,
                 tensor_shape: Union[torch.Size, List[int], Tuple[int]] = None,
                 scatter_gather_tensors: bool = False):
        """A helper schedule class for pipeline parallelism running environment.
        It uses interleaved 1F1B strategy. Other properties are similar as
        :class:`NonPipelineSchedule`.

        Args:
            num_microbatches (int): The number of microbatches.
            num_model_chunks (int): The number of model chunks.
            batch_data_process_func (Callable, optional):
                The preprocessing function which receives a batch of data, and it will be executed in `load_batch`.
            tensor_shape (torch.Size, optional): Specified shape in pipeline communication.
            scatter_gather_tensors (bool, optional):
                If set to `True`, communication will be reduced over pipeline when using 1D tensor parallelization.
        """
        assert num_microbatches % gpc.get_world_size(ParallelMode.PIPELINE) == 0, \
            'num_microbatches must be an integer multiple of pipeline parallel world size'
        super().__init__(num_microbatches,
                         batch_data_process_func=batch_data_process_func,
                         tensor_shape=tensor_shape,
                         scatter_gather_tensors=scatter_gather_tensors)
        gpc.set_virtual_pipeline_parallel_size(num_model_chunks)
        gpc.set_virtual_pipeline_parallel_rank(0)
        self.num_model_chunks = num_model_chunks

    def pre_processing(self, engine):
        if isinstance(engine.model, ShardedModelV2):
            self.dtype = torch.half
        elif isinstance(engine.model[0], NaiveAMPModel):
            self.dtype = torch.half
        for model in engine.model:
            if isinstance(model, NaiveAMPModel):
                model = model.model
            sig = inspect.signature(model.forward)
            for p in sig.parameters.values():
                assert p.kind != inspect.Parameter.VAR_POSITIONAL, '*args is not supported'

    def load_batch(self, data_iter):
        super().load_batch(data_iter)
        # overwrite microbatch_offset, since model chunks load the same microbatch, and should tract the offset
        self.microbatch_offset = [0 for _ in range(self.num_model_chunks)]

    def load_micro_batch(self, model_chunk_id):
        data = self._get_data_slice(self.batch_data, self.microbatch_offset[model_chunk_id])
        label = self._get_data_slice(self.batch_label, self.microbatch_offset[model_chunk_id])
        self.microbatch_offset[model_chunk_id] += self.microbatch_size
        return self._move_to_device(data), self._move_to_device(label)

    def forward_step(self,
                     engine,
                     model_chunk_id,
                     input_tensor,
                     return_tensors,
                     return_output_label=True,
                     accum_loss=None):
        """Forward step for passed-in model. If it is the first stage, the input tensor 
        is obtained from data_iterator, otherwise the passed-in input_tensor is used.
        Returns output tensor. This is a helper function and can be ignored by users.

        Args:
            engine (colossalai.engine.Engine): Colossalai engine for training and inference.
            model_chunk_id (int): The id of model chunks.
            input_tensor (:class:`torch.Tensor`): Input tensor for this pipeline stage.
            return_tensors (List[:class:`torch.Tensor`]): A list of tensors to return.
            return_output_label (bool, optional): Whether returns output labels.
            accum_loss (optional): Where accumulated loss stores.
        Returns:
            :class:`torch.Tensor`: output or the loss value of the current pipeline stage.
        """
        data, label = self.load_micro_batch(model_chunk_id)
        output_tensor = self._call_engine(engine.model[model_chunk_id], input_tensor, data)

        if gpc.is_pipeline_last_stage():
            if return_output_label:
                return_tensors.append((output_tensor, label))
            if accum_loss is not None:
                loss_reduced = self._call_engine_criterion(engine, output_tensor, label) / self.num_microbatches
                accum_loss.add_(loss_reduced.detach())
                return loss_reduced
            else:
                # forward only, it's useless since backward is not needed
                return output_tensor
        else:
            assert isinstance(
                output_tensor,
                torch.Tensor), 'Output of model using pipeline parallelism must be a tensor (except the last stage).'
            self._logger.debug(
                f'Global rank {gpc.get_global_rank()}, pipeline rank {gpc.get_local_rank(ParallelMode.PIPELINE)} forward output tensor {output_tensor.shape}, dtype {output_tensor.dtype}'
            )
            return output_tensor

    def forward_backward_step(self, engine, data_iter, forward_only=False, return_loss=True, return_output_label=True):
        """Run interleaved 1F1B schedule (model split into model chunks), with
        communication between pipeline stages as needed.

        Args:
            engine (colossalai.engine.Engine): Colossalai engine for training and inference.
            data_iter (Iterable): Dataloader as the form of an iterator, obtained by calling iter(dataloader).
            forward_only (bool, optional):
                Whether run forward step only. Default is false. If true, no backward will be run.
            return_loss (bool, optional): Whether returns the loss value. Default is true.
            return_output_label (bool, optional): If False, the output and label won't be returned.

        Returns:
            Tuple[:class:`torch.Tensor`]: A tuple of (output, label, loss), loss and label could be None.
                The loss would be returned only in the last stage.
        """
        assert forward_only or return_loss, \
            'The argument \'return_loss\' has to be True when \'forward_only\' is False, but got False.'
        self.load_batch(data_iter)
        model = engine.model
        input_tensors = [[] for _ in range(len(model))]
        output_tensors = [[] for _ in range(len(model))]
        return_tensors = []
        if not forward_only:
            output_tensor_grads = [[] for _ in range(len(model))]
        if return_loss and gpc.is_pipeline_last_stage(ignore_virtual=True):
            accum_loss = torch.zeros(1, device=get_current_device())
        else:
            accum_loss = None

        # Used for tensor meta information communication
        input_tensor_shapes = [self.tensor_shape for _ in range(len(model))]
        output_tensor_shapes = [None for _ in range(len(model))]
        send_tensor_shape_flags = [self.tensor_shape is None for _ in range(len(model))]

        pipeline_parallel_size = gpc.get_world_size(ParallelMode.PIPELINE)
        pipeline_parallel_rank = gpc.get_local_rank(ParallelMode.PIPELINE)

        # Compute number of warmup and remaining microbatches.
        num_model_chunks = len(model)
        num_microbatches = self.num_microbatches * num_model_chunks
        all_warmup_microbatches = False
        if forward_only:
            num_warmup_microbatches = num_microbatches
        else:
            # Run all forward passes and then all backward passes if number of
            # microbatches is just the number of pipeline stages.
            # Otherwise, perform (num_model_chunks-1)*pipeline_parallel_size on
            # all workers, followed by more microbatches after depending on
            # stage ID (more forward passes for earlier stages, later stages can
            # immediately start with 1F1B).
            if self.num_microbatches == pipeline_parallel_size:
                num_warmup_microbatches = num_microbatches
                all_warmup_microbatches = True
            else:
                num_warmup_microbatches = \
                    (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
                num_warmup_microbatches += (num_model_chunks - 1) * pipeline_parallel_size
                num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
        num_microbatches_remaining = \
            num_microbatches - num_warmup_microbatches

        def get_model_chunk_id(microbatch_id, forward):
            """Helper method to get the model chunk ID given the iteration number."""
            microbatch_id_in_group = microbatch_id % (pipeline_parallel_size * num_model_chunks)
            model_chunk_id = microbatch_id_in_group // pipeline_parallel_size
            if not forward:
                model_chunk_id = (num_model_chunks - model_chunk_id - 1)
            return model_chunk_id

        def forward_step_helper(microbatch_id):
            """Helper method to run forward step with model split into chunks
            (run set_virtual_pipeline_model_parallel_rank() before calling
            forward_step())."""
            model_chunk_id = get_model_chunk_id(microbatch_id, forward=True)
            gpc.set_virtual_pipeline_parallel_rank(model_chunk_id)

            # forward step
            if gpc.is_pipeline_first_stage():
                if len(input_tensors[model_chunk_id]) == \
                        len(output_tensors[model_chunk_id]):
                    input_tensors[model_chunk_id].append(None)
            input_tensor = input_tensors[model_chunk_id][-1]
            output_tensor = self.forward_step(engine,
                                              model_chunk_id,
                                              input_tensor,
                                              return_tensors,
                                              return_output_label=return_output_label,
                                              accum_loss=accum_loss)
            output_tensors[model_chunk_id].append(output_tensor)

            # if forward-only, no need to save tensors for a backward pass
            if forward_only:
                input_tensors[model_chunk_id].pop()
                output_tensors[model_chunk_id].pop()

            return output_tensor

        def backward_step_helper(microbatch_id):
            """Helper method to run backward step with model split into chunks
            (run set_virtual_pipeline_model_parallel_rank() before calling
            backward_step())."""
            model_chunk_id = get_model_chunk_id(microbatch_id, forward=False)
            gpc.set_virtual_pipeline_parallel_rank(model_chunk_id)

            if gpc.is_pipeline_last_stage():
                if len(output_tensor_grads[model_chunk_id]) == 0:
                    output_tensor_grads[model_chunk_id].append(None)
            input_tensor = input_tensors[model_chunk_id].pop(0)
            output_tensor = output_tensors[model_chunk_id].pop(0)
            output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)
            input_tensor_grad = self.backward_step(engine, input_tensor, output_tensor, output_tensor_grad)

            return input_tensor_grad

        # Run warmup forward passes.
        gpc.set_virtual_pipeline_parallel_rank(0)
        if not gpc.is_pipeline_first_stage():
            input_tensor_shapes[0] = comm.recv_tensor_meta(input_tensor_shapes[0])
        input_tensors[0].append(
            comm.recv_forward(input_tensor_shapes[0],
                              dtype=self.dtype,
                              scatter_gather_tensors=self.scatter_gather_tensors))

        for k in range(num_warmup_microbatches):
            model_chunk_id = get_model_chunk_id(k, forward=True)
            output_tensor = forward_step_helper(k)
            if not gpc.is_pipeline_last_stage():
                output_tensor_shapes[model_chunk_id] = output_tensor.shape
                send_tensor_shape_flags[model_chunk_id] = comm.send_tensor_meta(output_tensor,
                                                                                send_tensor_shape_flags[model_chunk_id])
            # Determine if tensor should be received from previous stage.
            next_forward_model_chunk_id = get_model_chunk_id(k + 1, forward=True)
            recv_prev = True
            if gpc.is_pipeline_first_stage(ignore_virtual=True):
                if next_forward_model_chunk_id == 0:
                    recv_prev = False
            if k == (num_microbatches - 1):
                recv_prev = False

            # Don't send tensor downstream if on last stage.
            if gpc.is_pipeline_last_stage():
                output_tensor = None

            with switch_virtual_pipeline_parallel_rank(next_forward_model_chunk_id):
                if not gpc.is_pipeline_first_stage():
                    input_tensor_shapes[next_forward_model_chunk_id] = comm.recv_tensor_meta(
                        input_tensor_shapes[next_forward_model_chunk_id])
            # Send and receive tensors as appropriate (send tensors computed
            # in this iteration; receive tensors for next iteration).
            input_shape = input_tensor_shapes[next_forward_model_chunk_id] if recv_prev else None
            if k == (num_warmup_microbatches - 1) and not forward_only and \
                    not all_warmup_microbatches:
                input_tensor_grad = None
                recv_next = True
                if gpc.is_pipeline_last_stage(ignore_virtual=True):
                    recv_next = False
                output_shape = output_tensor_shapes[num_model_chunks - 1] if recv_next else None
                input_tensor, output_tensor_grad = \
                    comm.send_forward_backward_recv_forward_backward(
                        output_tensor, input_tensor_grad,
                        input_shape,
                        output_shape,
                        recv_prev=recv_prev, recv_next=recv_next,
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors)
                output_tensor_grads[num_model_chunks - 1].append(output_tensor_grad)
            else:
                input_tensor = \
                    comm.send_forward_recv_forward(
                        output_tensor,
                        input_shape,
                        recv_prev=recv_prev,
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors)
            input_tensors[next_forward_model_chunk_id].append(input_tensor)

        # Run 1F1B in steady state.
        for k in range(num_microbatches_remaining):
            # Forward pass.
            forward_k = k + num_warmup_microbatches
            output_tensor = forward_step_helper(forward_k)

            # Backward pass.
            backward_k = k
            input_tensor_grad = backward_step_helper(backward_k)

            # Send output_tensor and input_tensor_grad, receive input_tensor
            # and output_tensor_grad.

            # Determine if current stage has anything to send in either direction,
            # otherwise set tensor to None.
            forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
            gpc.set_virtual_pipeline_parallel_rank(forward_model_chunk_id)
            if gpc.is_pipeline_last_stage():
                output_tensor = None

            backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
            gpc.set_virtual_pipeline_parallel_rank(backward_model_chunk_id)
            if gpc.is_pipeline_first_stage():
                input_tensor_grad = None

            # Determine if peers are sending, and where in data structure to put
            # received tensors.
            recv_prev = True
            if gpc.is_pipeline_first_stage(ignore_virtual=True):
                # First stage is ahead of last stage by (pipeline_parallel_size - 1).
                next_forward_model_chunk_id = get_model_chunk_id(forward_k - (pipeline_parallel_size - 1), forward=True)
                if next_forward_model_chunk_id == (num_model_chunks - 1):
                    recv_prev = False
                next_forward_model_chunk_id += 1
            else:
                next_forward_model_chunk_id = get_model_chunk_id(forward_k + 1, forward=True)

            recv_next = True
            if gpc.is_pipeline_last_stage(ignore_virtual=True):
                # Last stage is ahead of first stage by (pipeline_parallel_size - 1).
                next_backward_model_chunk_id = get_model_chunk_id(backward_k - (pipeline_parallel_size - 1),
                                                                  forward=False)
                if next_backward_model_chunk_id == 0:
                    recv_next = False
                next_backward_model_chunk_id -= 1
            else:
                next_backward_model_chunk_id = get_model_chunk_id(backward_k + 1, forward=False)

            # If last iteration, don't receive; we already received one extra
            # before the start of the for loop.
            if k == (num_microbatches_remaining - 1):
                recv_prev = False

            input_shape = input_tensor_shapes[next_forward_model_chunk_id] if recv_prev else None
            output_shape = output_tensor_shapes[next_backward_model_chunk_id] if recv_next else None
            # Communicate tensors.
            input_tensor, output_tensor_grad = \
                comm.send_forward_backward_recv_forward_backward(
                    output_tensor, input_tensor_grad,
                    input_shape,
                    output_shape,
                    recv_prev=recv_prev, recv_next=recv_next,
                    dtype=self.dtype,
                    scatter_gather_tensors=self.scatter_gather_tensors)

            # Put input_tensor and output_tensor_grad in data structures in the
            # right location.
            if recv_prev:
                input_tensors[next_forward_model_chunk_id].append(input_tensor)
            if recv_next:
                output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)

        # Run cooldown backward passes (flush out pipeline).
        if not forward_only:
            if all_warmup_microbatches:
                output_tensor_grads[num_model_chunks - 1].append(
                    comm.recv_backward(output_tensor_shapes[num_model_chunks - 1],
                                       scatter_gather_tensors=self.scatter_gather_tensors))
            for k in range(num_microbatches_remaining, num_microbatches):
                input_tensor_grad = backward_step_helper(k)
                next_backward_model_chunk_id = get_model_chunk_id(k + 1, forward=False)
                recv_next = True
                if gpc.is_pipeline_last_stage(ignore_virtual=True):
                    if next_backward_model_chunk_id == (num_model_chunks - 1):
                        recv_next = False
                if k == (num_microbatches - 1):
                    recv_next = False
                output_shape = output_tensor_shapes[next_backward_model_chunk_id] if recv_next else None
                output_tensor_grads[next_backward_model_chunk_id].append(
                    comm.send_backward_recv_backward(input_tensor_grad,
                                                     output_shape,
                                                     recv_next=recv_next,
                                                     dtype=self.dtype,
                                                     scatter_gather_tensors=self.scatter_gather_tensors))

        if len(return_tensors) > 0:
            output, label = pack_return_tensors(return_tensors)
            return output, label, accum_loss
        else:
            return None, None, accum_loss
