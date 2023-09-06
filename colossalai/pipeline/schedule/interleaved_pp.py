from functools import partial
from typing import Any, Callable, Iterable, List, Optional, Union

import torch
import torch.cuda
from torch.nn import Module
from torch.utils._pytree import tree_map

from colossalai.interface import OptimizerWrapper
from colossalai.pipeline.p2p import PipelineP2PCommunication
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.utils.cuda import get_current_device

from ._utils import detach, get_batch_size, get_micro_batch, merge_batch, model_forward, retain_grad, to_device
from .base import PipelineSchedule


class InterleavedSchedule(PipelineSchedule):

    def __init__(self, num_microbatches: int, num_model_chunks: int, stage_manager: PipelineStageManager) -> None:
        self.num_model_chunks = num_model_chunks
        assert num_microbatches % self.num_model_chunks == 0, \
            "Number of microbatches should be an integer multiple of number of model chunks"
        super().__init__(stage_manager)
        self.comm = PipelineP2PCommunication(stage_manager)
        self.num_microbatches = num_microbatches
        self.batch: Optional[Any] = None
        self.batch_size: Optional[int] = None
        self.microbatch_offset: Optional[int] = None
        self.microbatch_size: Optional[int] = None

    def load_batch(self, data_iter: Iterable, device: Optional[torch.device] = None) -> None:
        """Load a batch from data iterator.

        Args:
            data_iter (Iterable): Data iterator.
            device (Optional[torch.device], optional): Target device. Defaults to None.
        """
        batch = next(data_iter)
        if device is not None:
            batch = tree_map(partial(to_device, device=device), batch)
        self.batch = batch
        self.batch_size = get_batch_size(batch)
        self.microbatch_offset = [0 for _ in range(self.num_model_chunks)]
        assert self.batch_size % self.num_microbatches == 0, \
            "Batch size should divided by the number of microbatches"
        self.microbatch_size = self.batch_size // self.num_microbatches

    def load_micro_batch(self, model_chunk_id: int) -> Any:
        """Load a micro batch from the current batch.

        Args:
            microbatch_id (int): the current model chunk idx.

        Returns:
            Any: Micro batch.
        """
        micro_batch = get_micro_batch(self.batch, self.microbatch_offset[model_chunk_id], self.microbatch_size)
        self.microbatch_offset[model_chunk_id] += self.microbatch_size
        return tree_map(partial(to_device, device=get_current_device()), micro_batch)

    def get_model_chunk_id(self, microbatch_id: int, forward: bool) -> int:
        """Helper method to get the model chunk ID given the iteration number.

        Args:
            microbatch_id (int): the current microbatch idx
            forward (bool): if is the forward process

        Returns:
            int: The model chunk idx of the input microbatch_id
        """
        microbatch_id_in_group = (microbatch_id) % (self.stage_manager.num_stages * self.num_model_chunks)
        model_chunk_id = microbatch_id_in_group // self.stage_manager.num_stages
        if not forward:
            model_chunk_id = (self.num_model_chunks - model_chunk_id - 1)
        return model_chunk_id

    def is_first_stage(self, model_chunk_id: int) -> bool:
        """Is the current virtual stage the first stage

        Args:
            model_chunk_id (int): The current model chunk idx.

        Returns:
            bool: Whether the current virtual stage is the first stage.
        """
        if self.stage_manager.is_first_stage() and model_chunk_id == 0:
            return True
        return False

    def is_last_stage(self, model_chunk_id: int) -> bool:
        """Is the current virtual stage the last stage

        Args:
            model_chunk_id (int): The current model chunk idx.

        Returns:
            bool: Whether the current virtual stage is the last stage.
        """
        if self.stage_manager.is_last_stage() and model_chunk_id == self.num_model_chunks - 1:
            return True
        return False

    def recv_forward(self, model_chunk_id: int, prev_rank: int = None) -> Any:
        """Copy the forward output from the previous stage in pipeline as the input tensor of this stage.
           For interleaved 1F1B.

        Args:
            model_chunk_id (int): The current model chunk idx.
            prev_rank (int, optional): The rank of the source of the tensor.

        Returns:
            Any: The input tensor or input tensor list.
        """
        if self.is_first_stage(model_chunk_id):
            input_tensor = None
        else:
            input_tensor = self.comm.recv_forward(prev_rank)

        return input_tensor

    def recv_backward(self, model_chunk_id: int, next_rank: int = None) -> Any:
        """Copy the gradient tensor from the next stage in pipeline as the input gradient of this stage.
           For interleaved 1F1B.

        Args:
            model_chunk_id (int): The current model chunk idx.
            next_rank (int, optional): The rank of the source of the tensor.

        Returns:
            Any: The input gradient tensor or gradient tensor list.
        """
        if self.is_last_stage(model_chunk_id):
            output_tensor_grad = None
        else:
            output_tensor_grad = self.comm.recv_backward(next_rank)

        return output_tensor_grad

    def send_forward(self, model_chunk_id, output_object: Any, next_rank: int = None) -> None:
        """Sends the input tensor to the next stage in pipeline.
           For interleaved 1F1B.

        Args:
            model_chunk_id (int): The current model chunk idx.
            output_object (Any): Object to be sent.
            next_rank (int, optional): The rank of the recipient of the tensor.
        """
        if not self.is_last_stage(model_chunk_id):
            self.comm.send_forward(output_object, next_rank)

    def send_backward(self, model_chunk_id, input_object: Any, prev_rank: int = None) -> None:
        """Sends the gradient tensor to the previous stage in pipeline.
           For interleaved 1F1B.

        Args:
            model_chunk_id (int): The current model chunk idx.
            input_object (Any): Object to be sent.
            prev_rank (int, optional): The rank of the recipient of the tensor
        """
        if not self.is_first_stage(model_chunk_id):
            self.comm.send_backward(input_object, prev_rank)

    def forward_step(self,
                     model_chunk: Module,
                     model_chunk_id: int,
                     input_obj: Optional[dict],
                     criterion: Callable,
                     accum_loss: Optional[torch.Tensor] = None,
                     outputs: Optional[List[Any]] = None) -> Union[torch.Tensor, dict]:
        """Forward one step of the pipeline
        Args:
            model (Module): Model Chunk to be run
            input_obj (Optional[dict]): The output from the previous stage. If it is the first stage, the `input_obj` is None.
            criterion (Callable): Criterion to calculate loss.
            accum_loss (Optional[torch.Tensor], optional): Accumulated loss. Defaults to None.
            outputs (Optional[List[Any]], optional): List to store the output of the last stage (final output). Defaults to None.

        Returns:
            Union[torch.Tensor, dict]: The intermediate output (dict) of the current stage. If it is the last stage, the output is the loss (Tensor).
        """
        micro_batch = self.load_micro_batch(model_chunk_id=model_chunk_id)

        # for the first stage, input_obj is None
        # for the non-first stage, input_obj is the output of the previous stage and it's must be a dict
        output_obj = model_forward(model_chunk[model_chunk_id], micro_batch, input_obj)

        if self.is_last_stage(model_chunk_id):
            loss = criterion(output_obj, micro_batch) / self.num_microbatches
            if accum_loss is not None:
                accum_loss.add_(loss.detach())
            if outputs is not None:
                outputs.append(tree_map(detach, output_obj))
            return loss
        else:
            return output_obj

    def backward_step(self, optimizer: OptimizerWrapper, input_obj: Optional[dict],
                      output_obj: Union[dict, torch.Tensor], output_obj_grad: Optional[dict]) -> Optional[dict]:
        """Backward one step of the pipeline

        Args:
            optimizer (OptimizerWrapper): Optimizer to update the model
            input_obj (Optional[dict]): Output of the previous stage. If it is the first stage, the `input_obj` is None.
            output_obj (Union[dict, torch.Tensor]): Output of the current stage. If it is the last stage, the output is the loss (Tensor).
            output_obj_grad (dict): Gradient of the `output_obj`. If it is the last stage, the `output_obj_grad` is None.

        Returns:
            Optional[dict]: Gradient of the `input_obj`. If it is the first stage, the `input_obj_grad` is None.
        """

        # Retain the grad on the input_obj.
        tree_map(retain_grad, input_obj)

        # Backward pass.
        if output_obj_grad is None:
            optimizer.backward(output_obj)
        else:
            if "backward_tensor_keys" not in output_obj:
                for k, grad in output_obj_grad.items():
                    optimizer.backward_by_grad(output_obj[k], grad)
            else:
                for k, grad in output_obj_grad.items():
                    output_obj[k].grad = grad
                for k in output_obj["backward_tensor_keys"]:
                    tensor_to_backward = output_obj[k]
                    optimizer.backward_by_grad(tensor_to_backward, tensor_to_backward.grad)

        # Collect the grad of the input_obj.
        input_obj_grad = None
        if input_obj is not None:
            input_obj_grad = {}
            for k, v in input_obj.items():
                if isinstance(v, torch.Tensor) and v.grad is not None:
                    input_obj_grad[k] = v.grad
        return input_obj_grad

    def forward_backward_step(self,
                              model_chunk: Module,
                              data_iter: Iterable,
                              criterion: Callable[..., Any],
                              optimizer: Optional[OptimizerWrapper] = None,
                              return_loss: bool = False,
                              return_outputs: bool = False) -> dict:
        """Runs interleaved 1F1B schedule, with communication between pipeline stages.

        Args:
            model_chunk (List[Module]): Model Chunk to be trained.
            data_iter (Iterable): Data iterator.
            criterion (Callable[[Any, Any], Tensor]): Criterion to be used. It should take two arguments: model outputs and inputs, and returns loss tensor.
            optimizer (OptimizerWrapper, optional): Optimizer to be used. Can be None when only forward is executed. Defaults to None.
            return_loss (bool, optional): Whether to return loss. Defaults to False. Whether to return loss.
            return_outputs (bool, optional): Whether to return model outputs. Defaults to False. Whether to return model outputs.

        Returns:
            dict: A dict with keys: 'loss' and 'outputs'.
        """
        forward_only = not torch.is_grad_enabled()
        if optimizer is None:
            assert forward_only, "Optimizer should be passed when doing backward."

        self.load_batch(data_iter)
        num_model_chunks = len(model_chunk)

        # num_warmup_microbatches is the step when not all the processes are working
        num_microbatches = self.num_microbatches * num_model_chunks
        if forward_only:
            num_warmup_microbatches = num_microbatches
        else:
            num_warmup_microbatches = (self.stage_manager.num_stages - self.stage_manager.stage - 1) * 2
            num_warmup_microbatches += (num_model_chunks - 1) * self.stage_manager.num_stages
            num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)

        num_microbatches_remaining = num_microbatches - num_warmup_microbatches

        # Input, output tensors only need to be saved when doing backward passes
        input_objs = None
        output_objs = None

        if not forward_only:
            input_objs = [[] for _ in range(num_model_chunks)]
            output_objs = [[] for _ in range(num_model_chunks)]

        outputs = [] if return_outputs and self.stage_manager.is_last_stage() else None

        if return_loss and self.stage_manager.is_last_stage():
            accum_loss = torch.zeros(1, device=get_current_device())
        else:
            accum_loss = None

        # for ranks except the first one, get into recv state
        # print(self.stage_manager.stage,num_microbatches, num_warmup_microbatches, num_microbatches_remaining)
        input_obj = self.recv_forward(0)
        input_objs[0].append(input_obj)
        # Run warmup forward passes.
        for i in range(num_warmup_microbatches):
            model_chunk_id = self.get_model_chunk_id(i, forward=True)

            # recv first on first rank to avoid sending or recving at the same time
            if self.stage_manager.is_first_stage():
                input_obj = self.recv_forward(model_chunk_id)
                output_obj = self.forward_step(model_chunk, model_chunk_id, input_obj, criterion, accum_loss, outputs)
                self.send_forward(model_chunk_id, output_obj)
                if not forward_only:
                    input_objs[model_chunk_id].append(input_obj)
                    output_objs[model_chunk_id].append(output_obj)
            else:
                output_obj = self.forward_step(model_chunk, model_chunk_id, input_obj, criterion, accum_loss, outputs)
                if not forward_only:
                    output_objs[model_chunk_id].append(output_obj)
                self.send_forward(model_chunk_id, output_obj)
                if num_microbatches_remaining == 0 and i + 1 == num_warmup_microbatches:
                    break
                else:
                    model_chunk_id = self.get_model_chunk_id(i + 1, forward=True)

                    input_obj = self.recv_forward(model_chunk_id)
                    if not forward_only:
                        input_objs[model_chunk_id].append(input_obj)

        # Run 1F1B in steady state.
        for i in range(num_microbatches_remaining):
            model_chunk_id = self.get_model_chunk_id(i + num_warmup_microbatches, forward=True)
            last_iteration = (i == (num_microbatches_remaining - 1))

            output_obj = self.forward_step(model_chunk, model_chunk_id, input_obj, criterion, accum_loss, outputs)
            if forward_only:
                self.send_forward(model_chunk_id, output_obj)

                if not last_iteration:
                    input_obj = self.recv_forward(model_chunk_id)

            else:
                self.send_forward(model_chunk_id, output_obj)
                # Add input_obj and output_obj to end of list.
                input_objs[model_chunk_id].append(input_obj)
                output_objs[model_chunk_id].append(output_obj)

                model_chunk_id = self.get_model_chunk_id(i, forward=False)
                output_obj_grad = self.recv_backward(model_chunk_id)

                # Pop output_obj and output_obj from the start of the list for
                # the backward pass.
                input_obj = input_objs[model_chunk_id].pop(0)
                output_obj = output_objs[model_chunk_id].pop(0)

                # backward
                input_obj_grad = self.backward_step(optimizer, input_obj, output_obj, output_obj_grad)

                if last_iteration:
                    input_obj = None
                else:
                    model_chunk_id = self.get_model_chunk_id(i + num_warmup_microbatches + 1, forward=True)
                    input_obj = self.recv_forward(model_chunk_id)
                model_chunk_id = self.get_model_chunk_id(i, forward=False)
                self.send_backward(model_chunk_id, input_obj_grad)

        # Run cooldown backward passes.
        if not forward_only:
            for i in range(num_microbatches_remaining, num_microbatches):
                model_chunk_id = self.get_model_chunk_id(i, forward=False)
                # print(f"{self.stage_manager.stage}/{model_chunk_id}: {len(input_objs[model_chunk_id])} {len(output_objs[model_chunk_id])} {i}")
                input_obj = input_objs[model_chunk_id].pop(0)
                output_obj = output_objs[model_chunk_id].pop(0)

                output_obj_grad = self.recv_backward(model_chunk_id)
                input_obj_grad = self.backward_step(optimizer, input_obj, output_obj, output_obj_grad)
                self.send_backward(model_chunk_id, input_obj_grad)

        if outputs is not None:
            outputs = merge_batch(outputs)
        return {'loss': accum_loss, 'outputs': outputs}
