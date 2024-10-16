from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import torch
import torch.cuda
from torch.nn import Module
from torch.utils._pytree import tree_map

from colossalai.accelerator import get_accelerator
from colossalai.interface import ModelWrapper, OptimizerWrapper
from colossalai.pipeline.p2p import PipelineP2PCommunication, create_send_metadata
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.quantization.fp8 import cast_from_fp8_pipeline, cast_to_fp8_pipeline
from colossalai.utils import get_current_device

from ._utils import (
    detach,
    get_batch_size,
    get_micro_batch,
    merge_batch,
    model_forward,
    retain_grad,
    to_device,
    tree_map_hf,
)
from .base import PipelineSchedule


class OneForwardOneBackwardSchedule(PipelineSchedule):
    def __init__(
        self,
        stage_manager: PipelineStageManager,
        num_microbatches: Optional[int] = None,
        microbatch_size: Optional[int] = None,
        enable_metadata_cache: bool = True,
        fp8_communication: bool = False,
    ) -> None:
        """1F1B pipeline schedule.

        Args:
            stage_manager (PipelineStageManager): Pipeline stage manager
            num_microbatches (Optional[int], optional): The number of microbatches. If not provided, it will be derived from microbatch size. Defaults to None.
            microbatch_size (Optional[int], optional): Microbatch size. If num_microbatches is provided, this will be ignored. Defaults to None.
        """
        super().__init__(stage_manager)
        assert (
            num_microbatches is not None or microbatch_size is not None
        ), "Either num_microbatches or microbatch_size should be provided"

        self.comm = PipelineP2PCommunication(stage_manager, overlap_p2p=False)

        self.num_microbatches = num_microbatches
        self.microbatch_size = microbatch_size
        self.batch: Optional[Any] = None
        self.batch_size: Optional[int] = None
        self.last_batch_size: Optional[int] = None
        self.microbatch_offset: Optional[int] = None

        # P2PMeta cache
        self.enable_metadata_cache = enable_metadata_cache
        self.send_tensor_metadata = True
        self.send_grad_metadata = True
        self.tensor_metadata_recv = None
        self.grad_metadata_recv = None

        self.fp8_communication = fp8_communication

    def load_batch(self, data_iter: Iterable, device: Optional[torch.device] = None) -> None:
        """Load a batch from data iterator.

        Args:
            data_iter (Iterable): Data iterator.
            device (Optional[torch.device], optional): Target device. Defaults to None.
        """
        batch = next(data_iter)
        if device is not None:
            batch = tree_map(partial(to_device, device=device), batch)

        self.microbatch_offset = 0
        self.batch = batch
        self.batch_size = get_batch_size(batch)

        if self.microbatch_size is None:
            assert self.batch_size % self.num_microbatches == 0, "Batch size should divided by # microbatches"
            self.microbatch_size = self.batch_size // self.num_microbatches
        if self.num_microbatches is None:
            assert self.batch_size % self.microbatch_size == 0, "Batch size should divided by the microbatch size"
            self.num_microbatches = self.batch_size // self.microbatch_size

        if not self.forward_only:
            assert self.last_batch_size is None or self.last_batch_size == self.batch_size
            assert self.batch_size == self.microbatch_size * self.num_microbatches

            assert (
                self.num_microbatches >= self.stage_manager.num_stages
            ), "Number of microbatch should be larger than number of stages"

        if self.forward_only:
            self.num_microbatches = (self.batch_size - 1) // self.microbatch_size + 1
            # NOTE: disable metadata cache when batch size changes (not valid anymore)
            if self.batch_size != self.last_batch_size:
                self.enable_metadata_cache = False
                self.send_tensor_metadata = True
                self.send_grad_metadata = True
                self.tensor_metadata_recv = None
                self.grad_metadata_recv = None

        self.last_batch_size = self.batch_size

    def load_micro_batch(self) -> Any:
        """Load a micro batch from the current batch.

        Returns:
            Any: Micro batch.
        """
        assert self.microbatch_offset <= self.batch_size, "Microbatches exhausted"
        micro_batch = get_micro_batch(self.batch, self.microbatch_offset, self.microbatch_size)
        self.microbatch_offset += self.microbatch_size
        return tree_map(partial(to_device, device=get_accelerator().get_current_device()), micro_batch)

    def recv_forward(self, prev_rank: int = None) -> Any:
        """Copy the forward output from the previous stage in pipeline as the input tensor of this stage.
           For 1F1B.

        Args:
            prev_rank (int, optional): The rank of the source of the tensor.

        Returns:
            Any: The input tensor or input tensor list.
        """
        if not self.stage_manager.is_first_stage():
            input_tensor, _ = self.comm.recv_forward(prev_rank, metadata_recv=self.tensor_metadata_recv)
            if self.enable_metadata_cache and self.tensor_metadata_recv is None:
                self.tensor_metadata_recv = create_send_metadata(input_tensor)

            if self.fp8_communication:
                cast_from_fp8_pipeline(input_tensor)
            return input_tensor

    def recv_backward(self, next_rank: int = None) -> Any:
        """Copy the gradient tensor from the next stage in pipeline as the input gradient of this stage.
           For 1F1B.

        Args:
            next_rank (int, optional): The rank of the source of the tensor.

        Returns:
            Any: The input gradient tensor or gradient tensor list.
        """
        if not self.stage_manager.is_last_stage():
            output_tensor_grad, _ = self.comm.recv_backward(next_rank, metadata_recv=self.grad_metadata_recv)
            if self.fp8_communication:
                cast_from_fp8_pipeline(output_tensor_grad)
            if self.enable_metadata_cache and self.grad_metadata_recv is None:
                self.grad_metadata_recv = create_send_metadata(output_tensor_grad)

            return output_tensor_grad

    def send_forward(self, output_tensor: Any, next_rank: int = None) -> None:
        """Sends the input tensor to the next stage in pipeline.
           For 1F1B.

        Args:
            output_object (Any): Object to be sent.
            next_rank (int, optional): The rank of the recipient of the tensor.
        """
        if not self.stage_manager.is_last_stage():
            if self.fp8_communication:
                cast_to_fp8_pipeline(output_tensor)
            self.comm.send_forward(output_tensor, next_rank, send_metadata=self.send_tensor_metadata)
            self.send_tensor_metadata = not self.enable_metadata_cache

            if self.fp8_communication:
                cast_from_fp8_pipeline(output_tensor, del_metadata=False)

    def send_backward(self, input_tensor_grad: Any, prev_rank: int = None) -> None:
        """Sends the gradient tensor to the previous stage in pipeline.
           For 1F1B.

        Args:
            input_object (Any): Object to be sent.
            prev_rank (int, optional): The rank of the recipient of the tensor
        """
        if not self.stage_manager.is_first_stage():
            if self.fp8_communication:
                cast_to_fp8_pipeline(input_tensor_grad)
            self.comm.send_backward(input_tensor_grad, prev_rank, send_metadata=self.send_grad_metadata)
            self.send_grad_metadata = not self.enable_metadata_cache
            if self.fp8_communication:
                cast_from_fp8_pipeline(input_tensor_grad, del_metadata=False)

    def send_forward_recv_backward(self, output_tensor: Any, send_first: Optional[bool] = None) -> Any:
        """Sends the input tensor to the next stage and copy the gradient tensor from the next stage in pipeline.
           For 1F1B.

        Args:
            output_object (Any): Object to be sent.
            next_rank (int, optional): The rank of the recipient of the tensor.
        """
        if not self.stage_manager.is_last_stage():
            if not self.send_tensor_metadata and self.grad_metadata_recv is not None:
                send_first = None
            if self.fp8_communication:
                cast_to_fp8_pipeline(output_tensor)
            output_tensor_grad, _ = self.comm.send_forward_recv_backward(
                output_tensor,
                send_metadata=self.send_tensor_metadata,
                metadata_recv=self.grad_metadata_recv,
                send_first=send_first,
            )
            self.send_tensor_metadata = not self.enable_metadata_cache
            if self.enable_metadata_cache and self.grad_metadata_recv is None:
                self.grad_metadata_recv = create_send_metadata(output_tensor_grad)
            if self.fp8_communication:
                cast_from_fp8_pipeline(output_tensor, del_metadata=False)
                cast_from_fp8_pipeline(output_tensor_grad)

            return output_tensor_grad

    def send_backward_recv_forward(self, input_tensor_grad: Any, send_first: Optional[bool] = None) -> Any:
        """Sends the gradient tensor to the previous stage and copy the input tensor from the previous stage in pipeline.
           For 1F1B.

        Args:
            output_object (Any): Object to be sent.
            prev_rank (int, optional): The rank of the recipient of the tensor.
        """
        if not self.stage_manager.is_first_stage():
            if not self.send_grad_metadata and self.tensor_metadata_recv is not None:
                send_first = None  # must not fallback
            if self.fp8_communication:
                cast_to_fp8_pipeline(input_tensor_grad)
            input_tensor, _ = self.comm.send_backward_recv_forward(
                input_tensor_grad,
                send_metadata=self.send_grad_metadata,
                metadata_recv=self.tensor_metadata_recv,
                send_first=send_first,
            )
            self.send_grad_metadata = not self.enable_metadata_cache
            if self.enable_metadata_cache and self.tensor_metadata_recv is None:
                self.tensor_metadata_recv = create_send_metadata(input_tensor)
            if self.fp8_communication:
                cast_from_fp8_pipeline(input_tensor)
                cast_from_fp8_pipeline(input_tensor_grad, del_metadata=False)

            return input_tensor

    def forward_step(
        self,
        model: Module,
        input_obj: Optional[dict],
        criterion: Callable,
        accum_loss: Optional[torch.Tensor] = None,
        outputs: Optional[List[Any]] = None,
    ) -> Union[torch.Tensor, dict]:
        """Forward one step of the pipeline

        Args:
            model (Module): Model to be run
            input_obj (Optional[dict]): The output from the previous stage. If it is the first stage, the `input_obj` is None.
            criterion (Callable): Criterion to calculate loss.
            accum_loss (Optional[torch.Tensor], optional): Accumulated loss. Defaults to None.
            outputs (Optional[List[Any]], optional): List to store the output of the last stage (final output). Defaults to None.

        Returns:
            Union[torch.Tensor, dict]: The intermediate output (dict) of the current stage. If it is the last stage, the output is the loss (Tensor).
        """
        micro_batch = self.load_micro_batch()
        # for the first stage, input_obj is None
        # for the non-first stage, input_obj is the output of the previous stage and it's must be a dict
        output_obj = model_forward(model, micro_batch, input_obj)
        if self.stage_manager.is_last_stage():
            loss = criterion(output_obj, micro_batch) / self.num_microbatches

            if accum_loss is not None:
                accum_loss.add_(loss.data)
            if outputs is not None:
                outputs.append(tree_map_hf(detach, output_obj))
            return loss
        else:
            return output_obj

    def backward_step(
        self,
        optimizer: OptimizerWrapper,
        input_obj: Optional[dict],
        output_obj: Union[dict, torch.Tensor],
        output_obj_grad: Optional[dict],
    ) -> Optional[dict]:
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
            keys = output_obj.get("backward_tensor_keys", output_obj_grad.keys())
            tensors_to_backward = []
            grads_to_backward = []
            for k in keys:
                tensors_to_backward.append(output_obj[k])
                grads_to_backward.append(output_obj_grad[k])
            if len(tensors_to_backward) == 1:
                optimizer.backward_by_grad(tensors_to_backward[0], grads_to_backward[0])
            else:
                optimizer.backward_by_grad(tensors_to_backward, grads_to_backward)

        # Collect the grad of the input_obj.
        input_obj_grad = None
        if input_obj is not None:
            input_obj_grad = {}
            for k, v in input_obj.items():
                if isinstance(v, torch.Tensor) and v.grad is not None:
                    input_obj_grad[k] = v.grad
        return input_obj_grad

    def run_forward_only(
        self,
        model: Module,
        data_iter: Iterable,
        criterion: Callable[..., Any],
        return_loss: bool = False,
        return_outputs: bool = False,
    ) -> Dict:
        """
        Runs forward only schedule, with communication between pipeline stages.
        """
        assert self.forward_only

        self.load_batch(data_iter)

        accum_loss = None
        if return_loss and self.stage_manager.is_last_stage():
            accum_loss = torch.scalar_tensor(0, device=get_accelerator().get_current_device())
        outputs = [] if return_outputs and self.stage_manager.is_last_stage() else None

        for _ in range(self.num_microbatches):
            input_obj = self.recv_forward()
            output_obj = self.forward_step(model, input_obj, criterion, accum_loss, outputs)
            self.send_forward(output_obj)

        if outputs is not None:
            if isinstance(model, ModelWrapper):
                model = model.unwrap()
            batch_size_dim = getattr(model, "batch_size_dim", 0)
            outputs = merge_batch(outputs, batch_size_dim)
        return {"loss": accum_loss, "outputs": outputs}

    def run_forward_backward(
        self,
        model: Module,
        data_iter: Iterable,
        criterion: Callable[..., Any],
        optimizer: Optional[OptimizerWrapper] = None,
        return_loss: bool = False,
        return_outputs: bool = False,
    ) -> Dict:
        """
        Runs non-interleaved 1F1B schedule, with communication between pipeline stages.
        """
        assert not self.forward_only

        self.load_batch(data_iter)

        # num_warmup_microbatches is the step when not all the processes are working
        num_warmup_microbatches = self.stage_manager.num_stages - self.stage_manager.stage - 1
        num_warmup_microbatches = min(num_warmup_microbatches, self.num_microbatches)
        num_microbatches_remaining = self.num_microbatches - num_warmup_microbatches

        # Input, output tensors only need to be saved when doing backward passes
        input_objs, output_objs = [], []

        accum_loss = None
        if return_loss and self.stage_manager.is_last_stage():
            accum_loss = torch.scalar_tensor(0, device=get_current_device())
        outputs = [] if return_outputs and self.stage_manager.is_last_stage() else None

        # Run warmup forward passes.
        for i in range(num_warmup_microbatches):
            input_obj = self.recv_forward()
            output_obj = self.forward_step(model, input_obj, criterion, accum_loss, outputs)
            self.send_forward(output_obj)
            input_objs.append(input_obj)
            output_objs.append(output_obj)

        # Before running 1F1B, need to receive first forward tensor.
        # If all microbatches are run in warmup / cooldown phase, then no need to
        # receive this tensor here.
        if num_microbatches_remaining > 0:
            input_obj = self.recv_forward()

        # Run 1F1B in steady state.
        for i in range(num_microbatches_remaining):
            last_iteration = i == (num_microbatches_remaining - 1)

            output_obj = self.forward_step(model, input_obj, criterion, accum_loss, outputs)
            output_obj_grad = self.send_forward_recv_backward(output_obj, send_first=self.stage_manager.stage % 2 == 0)
            # Add input_obj and output_obj to end of list.
            input_objs.append(input_obj)
            output_objs.append(output_obj)

            # Pop output_obj and output_obj from the start of the list for
            # the backward pass.
            input_obj = input_objs.pop(0)
            output_obj = output_objs.pop(0)
            input_obj_grad = self.backward_step(optimizer, input_obj, output_obj, output_obj_grad)

            if last_iteration:
                self.send_backward(input_obj_grad)
            else:
                input_obj = self.send_backward_recv_forward(
                    input_obj_grad, send_first=self.stage_manager.stage % 2 == 0
                )

        # Run cooldown backward passes.
        for i in range(num_warmup_microbatches):
            input_obj = input_objs.pop(0)
            output_obj = output_objs.pop(0)

            output_obj_grad = self.recv_backward()
            input_obj_grad = self.backward_step(optimizer, input_obj, output_obj, output_obj_grad)
            self.send_backward(input_obj_grad)

        assert all(len(v) == 0 for v in input_objs) and all(len(v) == 0 for v in output_objs)

        if outputs is not None:
            if isinstance(model, ModelWrapper):
                model = model.unwrap()
            batch_size_dim = getattr(model, "batch_size_dim", 0)
            outputs = merge_batch(outputs, batch_size_dim)
        return {"loss": accum_loss, "outputs": outputs}

    def forward_backward_step(
        self,
        model: Module,
        data_iter: Iterable,
        criterion: Callable[..., Any],
        optimizer: Optional[OptimizerWrapper] = None,
        return_loss: bool = False,
        return_outputs: bool = False,
    ) -> dict:
        """
        Args:
            model (Module): Model to be trained.
            data_iter (Iterable): Data iterator.
            criterion (Callable[[Any, Any], Tensor]): Criterion to be used. It should take two arguments: model outputs and inputs, and returns loss tensor.
            optimizer (OptimizerWrapper, optional): Optimizer to be used. Can be None when only forward is executed. Defaults to None.
            return_loss (bool, optional): Whether to return loss. Defaults to False. Whether to return loss.
            return_outputs (bool, optional): Whether to return model outputs. Defaults to False. Whether to return model outputs.

        Returns:
            dict: Dictionary containing loss and outputs.
        """

        self.forward_only = not torch.is_grad_enabled()
        if optimizer is None:
            assert self.forward_only, "Optimizer should be passed when doing backward."

        if self.forward_only:
            result = self.run_forward_only(model, data_iter, criterion, return_loss, return_outputs)
        else:
            result = self.run_forward_backward(model, data_iter, criterion, optimizer, return_loss, return_outputs)

        return result
