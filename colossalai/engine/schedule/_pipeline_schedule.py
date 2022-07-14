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

from ._base_schedule import BaseSchedule


def get_tensor_shape():
    if hasattr(gpc.config, 'TENSOR_SHAPE'):
        return gpc.config.TENSOR_SHAPE

    if not gpc.is_initialized(ParallelMode.PIPELINE):
        return None

    if hasattr(gpc.config, 'SEQ_LENGTH') and hasattr(gpc.config, 'GLOBAL_BATCH_SIZE') and hasattr(
            gpc.config, 'GLOBAL_BATCH_SIZE') and hasattr(gpc.config, 'HIDDEN_SIZE'):
        if gpc.is_initialized(ParallelMode.DATA):
            dp_size = gpc.get_world_size(ParallelMode.DATA)
        else:
            dp_size = 1
        if gpc.is_initialized(ParallelMode.SEQUENCE):
            seq_size = gpc.get_world_size(ParallelMode.SEQUENCE)
        else:
            seq_size = 1

        tensor_shape = (gpc.config.SEQ_LENGTH // seq_size,
                        gpc.config.GLOBAL_BATCH_SIZE // dp_size // gpc.config.NUM_MICRO_BATCHES, gpc.config.HIDDEN_SIZE)
        return tensor_shape
    else:
        return None


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

    def __init__(self,
                 num_microbatches,
                 data_process_func: Callable = None,
                 tensor_shape: Union[torch.Size, List[int], Tuple[int]] = None,
                 scatter_gather_tensors: bool = False):

        # we need to make sure that the signature of the data_process_func is valid
        if data_process_func:
            sig = inspect.signature(data_process_func)
            assert len(sig.parameters) == 2, \
                'The data_process_func only takes in two parameters for NonPipelineSchedule, ' \
                'which is the tensors passed by the previous pipeline stage and the dataloader output from this stage, ' \
                'i.e. data_process_func(stage_output, dataloader_output).'

        super().__init__(data_process_func=data_process_func)

        assert num_microbatches > 0, f'expected num_microbatches to be larger then 1, but got {num_microbatches}'

        self.num_microbatches = num_microbatches
        self.dtype = torch.float
        assert not isinstance(tensor_shape,
                              int), "tensor_shape type should be one of Union[torch.Size, List[int], Tuple[int]]."
        if tensor_shape is None:
            self.tensor_shape = tensor_shape
        elif isinstance(tensor_shape, torch.Size):
            self.tensor_shape = tensor_shape
        else:
            self.tensor_shape = torch.Size(tensor_shape)
        self.scatter_gather_tensors = False
        if gpc.is_initialized(ParallelMode.PARALLEL_1D) and gpc.get_world_size(ParallelMode.PARALLEL_1D) > 1:
            self.scatter_gather_tensors = scatter_gather_tensors
        self._logger = get_dist_logger()

        # cache for the batch data
        self.batch_data = None

    def load_batch(self, data_iter):
        # Pipeline schedule just puts data in memory
        batch_data = super().load_batch(data_iter, to_gpu=False)
        self.microbatch_offset = 0
        assert self.batch_size % self.num_microbatches == 0, \
            "Batch size should divided by the number of microbatches"
        self.microbatch_size = self.batch_size // self.num_microbatches
        self.batch_data = batch_data

    def _get_data_slice(self, data, offset):
        if isinstance(data, torch.Tensor):
            return data[offset:offset + self.microbatch_size]
        elif isinstance(data, (list, tuple)):
            data_dict = {}
            for element in data:
                if isinstance(element, dict):
                    data_dict.update({k: v[offset:offset + self.microbatch_size] for k, v in element.items()})
                elif data_dict:
                    data_dict['label'] = element[offset:offset + self.microbatch_size]
            if data_dict:
                return data_dict
            return [val[offset:offset + self.microbatch_size] for val in data]
        elif isinstance(data, dict):
            return {k: v[offset:offset + self.microbatch_size] for k, v in data.items()}
        else:
            raise TypeError(f"Expected data to be of type torch.Tensor, list, tuple, or dict, but got {type(data)}")

    def load_micro_batch(self):
        mciro_batch_data = self._get_data_slice(self.batch_data, self.microbatch_offset)
        self.microbatch_offset += self.microbatch_size
        return self._move_to_device(mciro_batch_data)

    def pre_processing(self, engine):
        from colossalai.zero.sharded_model.sharded_model_v2 import ShardedModelV2
        # TODO: remove this after testing new zero with pipeline parallelism
        model = engine.model
        if isinstance(model, NaiveAMPModel):
            self.dtype = torch.half
            model = model.model
        if isinstance(model, ShardedModelV2):
            self.dtype = torch.half
            model = model.module
        # sig = inspect.signature(model.forward)
        # for p in sig.parameters.values():
        #     assert p.kind != inspect.Parameter.VAR_POSITIONAL, '*args is not supported'

    @staticmethod
    def _call_engine(model, data):
        if data is not None:
            if isinstance(data, torch.Tensor):
                return model(data)
            elif isinstance(data, (list, tuple)):
                return model(*data)
            elif isinstance(data, dict):
                stage_output = None
                if 'stage_output' in data:
                    stage_output = data.pop('stage_output')
                if stage_output is None:
                    return model(**data)
                elif isinstance(stage_output, torch.Tensor):
                    return model(stage_output, **data)
                elif isinstance(stage_output, (tuple, list)):
                    return model(*stage_output, **data)
                else:
                    raise TypeError(
                        f"Expected stage_output to be of type torch.Tensor, list, or tuple, but got {type(stage_output)}"
                    )
            else:
                raise TypeError(f"Expected data to be of type torch.Tensor, list, tuple, or dict, but got {type(data)}")

    def _get_actual_forward_func(self, module):
        if isinstance(module, NaiveAMPModel):
            sig = inspect.signature(module.model.forward)
        elif hasattr(module, 'colo_attr'):
            sig = inspect.signature(module.module.forward)
        else:
            sig = inspect.signature(module.forward)
        return sig

    def _get_data_label_for_current_step(self, stage_output, micro_batch_data, criterion, model):
        if self.data_process_func:
            # use customized function to get data and label
            data, label = self.data_process_func(stage_output, micro_batch_data)
        else:
            if isinstance(micro_batch_data, (tuple, list)):
                if gpc.is_first_rank(ParallelMode.PIPELINE):
                    # for the first stage, we use the data from the
                    # dataloader output by default
                    data, label = micro_batch_data
                else:
                    # for non-first stage, we use the output passed
                    # by the previous as the model input
                    data = stage_output
                    _, label = micro_batch_data
            elif isinstance(micro_batch_data, dict):
                data = {}
                data['stage_output'] = stage_output
                if 'label' in micro_batch_data:
                    label = micro_batch_data.pop('label')
                else:
                    label = None
                load_data = micro_batch_data
                data.update(load_data)
        return data, label

    def _forward_step(self, engine, input_obj, return_tensors, return_output_label=True, accum_loss=None):
        """Forward step for passed-in model. If it is the first stage, the input tensor 
        is obtained from data_iterator, otherwise the passed-in input_obj is used.
        Returns output tensor. This is a helper function and can be ignored by users.

        Args:
            engine (colossalai.engine.Engine): Colossalai engine for training and inference.
            input_obj (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): Input tensor for this pipeline stage.
            return_tensors (List[:class:`torch.Tensor`]): A list of tensors to return.
            return_output_label (bool, optional): Whether returns output labels.
            accum_loss (optional): Where accumulated loss stores.
        Returns:
            Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]: output or the loss value of the current pipeline stage.
        """
        micro_batch_data = self.load_micro_batch()

        data, label = self._get_data_label_for_current_step(input_obj, micro_batch_data, engine.criterion, engine.model)

        output_obj = self._call_engine(engine.model, data)

        if gpc.is_last_rank(ParallelMode.PIPELINE):
            if return_output_label:
                return_tensors.append((output_obj, label))
            if accum_loss is not None:
                loss_reduced = self._call_engine_criterion(engine, output_obj, label) / self.num_microbatches
                accum_loss.add_(loss_reduced.detach())
                return loss_reduced
            else:
                # forward only, it's useless since backward is not needed
                return output_obj
        else:
            if isinstance(output_obj, torch.Tensor):
                self._logger.debug(
                    f'Global rank {gpc.get_global_rank()}, pipeline rank {gpc.get_local_rank(ParallelMode.PIPELINE)} forward output tensor {output_obj.shape}, dtype {output_obj.dtype}'
                )
            return output_obj

    def _backward_step(self, engine, input_obj, output_obj, output_obj_grad):
        """Backward step through the passed-in output tensor. If it is the last stage, the 
        output_obj_grad is None, otherwise it is the gradients with respect to stage's output tensor.
        Returns the gradients with respect to the input tensor (None if first stage).
        This is a helper function and can be ignored by users.

        Args:
            engine (colossalai.engine.Engine): Colossalai engine for training and inference.
            input_obj (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): input tensor for this pipeline stage.
            output_obj (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): output tensor for this pipeline stage.
            output_obj_grad (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): gradient of output tensor for this pipeline stage.

        Returns:
            Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]: gradient of input tensor.
        """

        # Retain the grad on the input_obj.
        if input_obj is not None:
            if isinstance(input_obj, torch.Tensor):
                input_obj.retain_grad()
            else:
                for in_tensor in input_obj:
                    if in_tensor is not None:
                        in_tensor.retain_grad()
        # Backward pass.
        if output_obj_grad is None:
            engine.backward(output_obj)
        else:
            engine.backward_by_grad(output_obj, output_obj_grad)

        # Collect the grad of the input_obj.
        input_obj_grad = None
        if input_obj is not None:
            if isinstance(input_obj, torch.Tensor):
                input_obj_grad = input_obj.grad
            else:
                input_obj_grad = []
                for in_tensor in input_obj:
                    input_obj_grad.append(in_tensor.grad)

        return input_obj_grad

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
        input_objs = None
        output_objs = None
        if not forward_only:
            input_objs = []
            output_objs = []
        return_tensors = []
        if return_loss and gpc.is_pipeline_last_stage(ignore_virtual=True):
            accum_loss = torch.zeros(1, device=get_current_device())
        else:
            accum_loss = None
        # Used for tensor meta information communication
        ft_shapes = self.tensor_shape
        bt_shapes = None
        fs_checker = self.tensor_shape is None

        # Run warmup forward passes.
        for i in range(num_warmup_microbatches):
            if not gpc.is_first_rank(ParallelMode.PIPELINE):
                ft_shapes = comm.recv_obj_meta(ft_shapes)
            input_obj = comm.recv_forward(ft_shapes,
                                          dtype=self.dtype,
                                          scatter_gather_tensors=self.scatter_gather_tensors)
            output_obj = self._forward_step(engine,
                                            input_obj,
                                            return_tensors,
                                            return_output_label=return_output_label,
                                            accum_loss=accum_loss)
            if not gpc.is_last_rank(ParallelMode.PIPELINE):
                if isinstance(output_obj, torch.Tensor):
                    bt_shapes = output_obj.shape
                else:
                    bt_shapes = []
                    for out_tensor in output_obj:
                        bt_shapes.append(out_tensor.shape)
                fs_checker = comm.send_obj_meta(output_obj, fs_checker)
            comm.send_forward(output_obj, scatter_gather_tensors=self.scatter_gather_tensors)

            if not forward_only:
                input_objs.append(input_obj)
                output_objs.append(output_obj)

        # Before running 1F1B, need to receive first forward tensor.
        # If all microbatches are run in warmup / cooldown phase, then no need to
        # receive this tensor here.
        if num_microbatches_remaining > 0:
            if not gpc.is_first_rank(ParallelMode.PIPELINE):
                ft_shapes = comm.recv_obj_meta(ft_shapes)
            input_obj = comm.recv_forward(ft_shapes,
                                          dtype=self.dtype,
                                          scatter_gather_tensors=self.scatter_gather_tensors)

        # Run 1F1B in steady state.
        for i in range(num_microbatches_remaining):
            last_iteration = (i == (num_microbatches_remaining - 1))

            output_obj = self._forward_step(engine,
                                            input_obj,
                                            return_tensors,
                                            return_output_label=return_output_label,
                                            accum_loss=accum_loss)
            if forward_only:
                comm.send_forward(output_obj, scatter_gather_tensors=self.scatter_gather_tensors)

                if not last_iteration:
                    input_obj = comm.recv_forward(ft_shapes,
                                                  dtype=self.dtype,
                                                  scatter_gather_tensors=self.scatter_gather_tensors)

            else:
                output_obj_grad = comm.send_forward_recv_backward(output_obj,
                                                                  bt_shapes,
                                                                  dtype=self.dtype,
                                                                  scatter_gather_tensors=self.scatter_gather_tensors)

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
                    comm.send_backward(input_obj_grad, scatter_gather_tensors=self.scatter_gather_tensors)
                else:
                    input_obj = comm.send_backward_recv_forward(input_obj_grad,
                                                                ft_shapes,
                                                                dtype=self.dtype,
                                                                scatter_gather_tensors=self.scatter_gather_tensors)

        # Run cooldown backward passes.
        if not forward_only:
            for i in range(num_warmup_microbatches):
                input_obj = input_objs.pop(0)
                output_obj = output_objs.pop(0)

                output_obj_grad = comm.recv_backward(bt_shapes,
                                                     dtype=self.dtype,
                                                     scatter_gather_tensors=self.scatter_gather_tensors)

                input_obj_grad = self._backward_step(engine, input_obj, output_obj, output_obj_grad)

                comm.send_backward(input_obj_grad, scatter_gather_tensors=self.scatter_gather_tensors)

        if len(return_tensors) > 0:
            output, label = pack_return_tensors(return_tensors)
            return output, label, accum_loss
        else:
            return None, None, accum_loss


class InterleavedPipelineSchedule(PipelineSchedule):

    def __init__(self,
                 num_microbatches: int,
                 num_model_chunks: int,
                 data_process_func: Callable = None,
                 tensor_shape: Union[torch.Size, List[int], Tuple[int]] = None,
                 scatter_gather_tensors: bool = False):
        """A helper schedule class for pipeline parallelism running environment.
        It uses interleaved 1F1B strategy. Other properties are similar as
        :class:`NonPipelineSchedule`.

        Args:
            num_microbatches (int): The number of microbatches.
            num_model_chunks (int): The number of model chunks.
            data_process_func (Callable, optional):
                The preprocessing function which receives a batch of data, and it will be executed in `load_batch`.
            tensor_shape (torch.Size, optional): Specified shape in pipeline communication.
            scatter_gather_tensors (bool, optional):
                If set to `True`, communication will be reduced over pipeline when using 1D tensor parallelization.
        """
        assert num_microbatches % gpc.get_world_size(ParallelMode.PIPELINE) == 0, \
            'num_microbatches must be an integer multiple of pipeline parallel world size'
        assert isinstance(num_model_chunks, int) and num_model_chunks > 0, \
            f'expected num_model_chunks to be an integer and larger than 0, but got {num_model_chunks}'
        super().__init__(num_microbatches,
                         data_process_func=data_process_func,
                         tensor_shape=tensor_shape,
                         scatter_gather_tensors=scatter_gather_tensors)
        gpc.set_virtual_pipeline_parallel_size(num_model_chunks)
        gpc.set_virtual_pipeline_parallel_rank(0)
        self.num_model_chunks = num_model_chunks

    def pre_processing(self, engine):
        from colossalai.zero.sharded_model.sharded_model_v2 import ShardedModelV2
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
        self.microbatch_offset[model_chunk_id] += self.microbatch_size
        return self._move_to_device(data)

    def _forward_step(self,
                      engine,
                      model_chunk_id,
                      input_obj,
                      return_tensors,
                      return_output_label=True,
                      accum_loss=None):
        """Forward step for passed-in model. If it is the first stage, the input tensor 
        is obtained from data_iterator, otherwise the passed-in input_obj is used.
        Returns output tensor. This is a helper function and can be ignored by users.

        Args:
            engine (colossalai.engine.Engine): Colossalai engine for training and inference.
            model_chunk_id (int): The id of model chunks.
            input_obj (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): Input tensor for this pipeline stage.
            return_tensors (List[:class:`torch.Tensor`]): A list of tensors to return.
            return_output_label (bool, optional): Whether returns output labels.
            accum_loss (optional): Where accumulated loss stores.
        Returns:
            Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]: output or the loss value of the current pipeline stage.
        """
        micro_batch_data = self.load_micro_batch(model_chunk_id)
        data, label = self._get_data_label_for_current_step(input_obj, micro_batch_data, engine.criterion,
                                                            engine.model[model_chunk_id])

        output_obj = self._call_engine(engine.model[model_chunk_id], data)

        if gpc.is_pipeline_last_stage():
            if return_output_label:
                return_tensors.append((output_obj, label))
            if accum_loss is not None:
                loss_reduced = self._call_engine_criterion(engine, output_obj, label) / self.num_microbatches
                accum_loss.add_(loss_reduced.detach())
                return loss_reduced
            else:
                # forward only, it's useless since backward is not needed
                return output_obj
        else:
            if isinstance(output_obj, torch.Tensor):
                self._logger.debug(
                    f'Global rank {gpc.get_global_rank()}, pipeline rank {gpc.get_local_rank(ParallelMode.PIPELINE)} forward output tensor {output_obj.shape}, dtype {output_obj.dtype}'
                )
            return output_obj

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
        input_objs = [[] for _ in range(len(model))]
        output_objs = [[] for _ in range(len(model))]
        return_tensors = []
        if not forward_only:
            output_obj_grads = [[] for _ in range(len(model))]
        if return_loss and gpc.is_pipeline_last_stage(ignore_virtual=True):
            accum_loss = torch.zeros(1, device=get_current_device())
        else:
            accum_loss = None

        # Used for obj meta information communication
        input_obj_shapes = [self.tensor_shape for _ in range(len(model))]
        output_obj_shapes = [None for _ in range(len(model))]
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

        def _forward_step_helper(microbatch_id):
            """Helper method to run forward step with model split into chunks
            (run set_virtual_pipeline_model_parallel_rank() before calling
            forward_step())."""
            model_chunk_id = get_model_chunk_id(microbatch_id, forward=True)
            gpc.set_virtual_pipeline_parallel_rank(model_chunk_id)

            # forward step
            if gpc.is_pipeline_first_stage():
                if len(input_objs[model_chunk_id]) == \
                        len(output_objs[model_chunk_id]):
                    input_objs[model_chunk_id].append(None)
            input_obj = input_objs[model_chunk_id][-1]
            output_obj = self._forward_step(engine,
                                            model_chunk_id,
                                            input_obj,
                                            return_tensors,
                                            return_output_label=return_output_label,
                                            accum_loss=accum_loss)
            output_objs[model_chunk_id].append(output_obj)

            # if forward-only, no need to save tensors for a backward pass
            if forward_only:
                input_objs[model_chunk_id].pop()
                output_objs[model_chunk_id].pop()

            return output_obj

        def _backward_step_helper(microbatch_id):
            """Helper method to run backward step with model split into chunks
            (run set_virtual_pipeline_model_parallel_rank() before calling
            backward_step())."""
            model_chunk_id = get_model_chunk_id(microbatch_id, forward=False)
            gpc.set_virtual_pipeline_parallel_rank(model_chunk_id)

            if gpc.is_pipeline_last_stage():
                if len(output_obj_grads[model_chunk_id]) == 0:
                    output_obj_grads[model_chunk_id].append(None)
            input_obj = input_objs[model_chunk_id].pop(0)
            output_obj = output_objs[model_chunk_id].pop(0)
            output_obj_grad = output_obj_grads[model_chunk_id].pop(0)
            input_obj_grad = self._backward_step(engine, input_obj, output_obj, output_obj_grad)

            return input_obj_grad

        # Run warmup forward passes.
        gpc.set_virtual_pipeline_parallel_rank(0)
        if not gpc.is_pipeline_first_stage():
            input_obj_shapes[0] = comm.recv_obj_meta(input_obj_shapes[0])
        input_objs[0].append(
            comm.recv_forward(input_obj_shapes[0], dtype=self.dtype,
                              scatter_gather_tensors=self.scatter_gather_tensors))

        for k in range(num_warmup_microbatches):
            model_chunk_id = get_model_chunk_id(k, forward=True)
            output_obj = _forward_step_helper(k)
            if not gpc.is_pipeline_last_stage():
                if isinstance(output_obj, torch.Tensor):
                    output_obj_shapes[model_chunk_id] = output_obj.shape
                else:
                    output_obj_shapes[model_chunk_id] = []
                    for out_tensor in output_obj:
                        output_obj_shapes[model_chunk_id].append(out_tensor.shape)
                send_tensor_shape_flags[model_chunk_id] = comm.send_obj_meta(output_obj,
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
                output_obj = None

            with switch_virtual_pipeline_parallel_rank(next_forward_model_chunk_id):
                if not gpc.is_pipeline_first_stage():
                    input_obj_shapes[next_forward_model_chunk_id] = comm.recv_obj_meta(
                        input_obj_shapes[next_forward_model_chunk_id])
            # Send and receive tensors as appropriate (send tensors computed
            # in this iteration; receive tensors for next iteration).
            input_shape = input_obj_shapes[next_forward_model_chunk_id] if recv_prev else None
            if k == (num_warmup_microbatches - 1) and not forward_only and \
                    not all_warmup_microbatches:
                input_obj_grad = None
                recv_next = True
                if gpc.is_pipeline_last_stage(ignore_virtual=True):
                    recv_next = False
                output_shape = output_obj_shapes[num_model_chunks - 1] if recv_next else None
                input_obj, output_obj_grad = \
                    comm.send_forward_backward_recv_forward_backward(
                        output_obj, input_obj_grad,
                        input_shape,
                        output_shape,
                        recv_prev=recv_prev, recv_next=recv_next,
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors)
                output_obj_grads[num_model_chunks - 1].append(output_obj_grad)
            else:
                input_obj = \
                    comm.send_forward_recv_forward(
                        output_obj,
                        input_shape,
                        recv_prev=recv_prev,
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors)
            input_objs[next_forward_model_chunk_id].append(input_obj)

        # Run 1F1B in steady state.
        for k in range(num_microbatches_remaining):
            # Forward pass.
            forward_k = k + num_warmup_microbatches
            output_obj = _forward_step_helper(forward_k)

            # Backward pass.
            backward_k = k
            input_obj_grad = _backward_step_helper(backward_k)

            # Send output_obj and input_obj_grad, receive input_obj
            # and output_obj_grad.

            # Determine if current stage has anything to send in either direction,
            # otherwise set obj to None.
            forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
            gpc.set_virtual_pipeline_parallel_rank(forward_model_chunk_id)
            if gpc.is_pipeline_last_stage():
                output_obj = None

            backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
            gpc.set_virtual_pipeline_parallel_rank(backward_model_chunk_id)
            if gpc.is_pipeline_first_stage():
                input_obj_grad = None

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

            input_shape = input_obj_shapes[next_forward_model_chunk_id] if recv_prev else None
            output_shape = output_obj_shapes[next_backward_model_chunk_id] if recv_next else None
            # Communicate objs.
            input_obj, output_obj_grad = \
                comm.send_forward_backward_recv_forward_backward(
                    output_obj, input_obj_grad,
                    input_shape,
                    output_shape,
                    recv_prev=recv_prev, recv_next=recv_next,
                    dtype=self.dtype,
                    scatter_gather_tensors=self.scatter_gather_tensors)

            # Put input_obj and output_obj_grad in data structures in the
            # right location.
            if recv_prev:
                input_objs[next_forward_model_chunk_id].append(input_obj)
            if recv_next:
                output_obj_grads[next_backward_model_chunk_id].append(output_obj_grad)

        # Run cooldown backward passes (flush out pipeline).
        if not forward_only:
            if all_warmup_microbatches:
                output_obj_grads[num_model_chunks - 1].append(
                    comm.recv_backward(output_obj_shapes[num_model_chunks - 1],
                                       scatter_gather_tensors=self.scatter_gather_tensors))
            for k in range(num_microbatches_remaining, num_microbatches):
                input_obj_grad = _backward_step_helper(k)
                next_backward_model_chunk_id = get_model_chunk_id(k + 1, forward=False)
                recv_next = True
                if gpc.is_pipeline_last_stage(ignore_virtual=True):
                    if next_backward_model_chunk_id == (num_model_chunks - 1):
                        recv_next = False
                if k == (num_microbatches - 1):
                    recv_next = False
                output_shape = output_obj_shapes[next_backward_model_chunk_id] if recv_next else None
                output_obj_grads[next_backward_model_chunk_id].append(
                    comm.send_backward_recv_backward(input_obj_grad,
                                                     output_shape,
                                                     recv_next=recv_next,
                                                     dtype=self.dtype,
                                                     scatter_gather_tensors=self.scatter_gather_tensors))

        if len(return_tensors) > 0:
            output, label = pack_return_tensors(return_tensors)
            return output, label, accum_loss
        else:
            return None, None, accum_loss
