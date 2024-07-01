from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.cuda
import torch.distributed
from torch.nn import Module, ModuleList
from torch.utils._pytree import tree_map

from colossalai.accelerator import get_accelerator
from colossalai.interface import OptimizerWrapper
from colossalai.pipeline.p2p import PipelineP2PCommunication, create_send_metadata
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.utils import get_current_device

from ._utils import detach, get_batch_size, get_micro_batch, merge_batch, model_forward, retain_grad, to_device
from .base import PipelineSchedule


def _wait_p2p(wait_handles: List[torch.cuda.Event]) -> None:
    if wait_handles is not None:
        for req in wait_handles:
            req.wait()


class InterleavedSchedule(PipelineSchedule):
    def __init__(
        self,
        stage_manager: PipelineStageManager,
        num_model_chunks: int,
        num_microbatch: Optional[int] = None,
        microbatch_size: Optional[int] = None,
        enable_metadata_cache: bool = True,
        overlap_p2p: bool = True,
    ) -> None:
        super().__init__(stage_manager)
        assert (
            num_microbatch is not None or microbatch_size is not None
        ), "Either num_microbatch or microbatch_size should be provided"

        self.comm = PipelineP2PCommunication(stage_manager, overlap_p2p=overlap_p2p)
        self.overlap_p2p = overlap_p2p
        self.num_microbatch = num_microbatch
        self.microbatch_size = microbatch_size
        self.num_model_chunks = num_model_chunks

        self.batch: Any
        self.batch_size: int
        self.last_batch_size: Optional[int] = None
        self.microbatch_offset: List[int]

        # P2PMeta cache
        self.enable_metadata_cache = enable_metadata_cache
        self.send_tensor_metadata = True
        self.send_grad_metadata = True
        self.tensor_metadata_recv = None
        self.grad_metadata_recv = None

    def load_batch(self, data_iter: Iterable, device: Optional[torch.device] = None) -> None:
        """Load a batch from data iterator.

        Args:
            data_iter (Iterable): Data iterator.
            device (Optional[torch.device], optional): Target device. Defaults to None.
        """
        batch = next(data_iter)
        if device is not None:
            batch = tree_map(partial(to_device, device=device), batch)

        self.microbatch_offset = [0 for _ in range(self.num_model_chunks)]
        self.batch = batch
        self.batch_size = get_batch_size(batch)

        if self.microbatch_size is None:
            assert self.batch_size % self.num_microbatch == 0, "Batch size should divided by the number of microbatch"
            self.microbatch_size = self.batch_size // self.num_microbatch
        if self.num_microbatch is None:
            assert self.batch_size % self.microbatch_size == 0, "Batch size should divided by the microbatch size"
            self.num_microbatch = self.batch_size // self.microbatch_size

        if not self.forward_only:
            assert self.last_batch_size is None or self.last_batch_size == self.batch_size
            assert self.batch_size == self.microbatch_size * self.num_microbatch

            assert (
                self.num_microbatch % self.stage_manager.num_stages == 0
            ), "Number of microbatch should be an integer multiple of number of pipeline parallel devices"

        if self.forward_only:
            self.num_microbatch = (self.batch_size - 1) // self.microbatch_size + 1
            # NOTE: disable metadata cache when batch size changes (not valid anymore)
            if self.batch_size != self.last_batch_size:
                self.enable_metadata_cache = False
                self.send_tensor_metadata = True
                self.send_grad_metadata = True
                self.tensor_metadata_recv = None
                self.grad_metadata_recv = None

        self.last_batch_size = self.batch_size

    def load_micro_batch(self, model_chunk_id: int) -> Any:
        """Load a micro batch from the current batch.

        Args:
            microbatch_id (int): the current model chunk idx.

        Returns:
            Any: Micro batch.
        """
        assert self.microbatch_offset[model_chunk_id] <= self.batch_size, "Microbatches exhausted"
        micro_batch = get_micro_batch(self.batch, self.microbatch_offset[model_chunk_id], self.microbatch_size)
        self.microbatch_offset[model_chunk_id] += self.microbatch_size
        return tree_map(partial(to_device, device=get_accelerator().get_current_device()), micro_batch)

    def get_model_chunk_id(self, microbatch_id: int, is_forward: bool) -> int:
        """Helper method to get the model chunk ID given the iteration number.

        Args:
            microbatch_id (int): the current microbatch idx
            forward (bool): if is the forward process

        Returns:
            int: The model chunk idx of the input microbatch_id
        """
        assert (
            microbatch_id < self.num_microbatch * self.num_model_chunks
        ), f"microbatch_id {microbatch_id} is out of range ({self.num_microbatch * self.num_model_chunks})"
        microbatch_id_in_group = microbatch_id % (self.stage_manager.num_stages * self.num_model_chunks)
        model_chunk_id = microbatch_id_in_group // self.stage_manager.num_stages
        if not is_forward:
            # Reverse order
            model_chunk_id = self.num_model_chunks - model_chunk_id - 1
        return model_chunk_id

    def recv_forward(self, model_chunk_id: int, prev_rank: int = None) -> Tuple[Any, List]:
        """Copy the forward output from the previous stage in pipeline as the input tensor of this stage.
           For interleaved 1F1B.

        Args:
            model_chunk_id (int): The current model chunk idx.
            prev_rank (int, optional): The rank of the source of the tensor.

        Returns:
            Any: The input tensor or input tensor list.
            Any: The wait handles for the communication.
        """
        with self.stage_manager.switch_model_chunk_id(model_chunk_id):
            if not self.stage_manager.is_first_stage():
                input_tensor, wait_handles = self.comm.recv_forward(prev_rank, metadata_recv=self.tensor_metadata_recv)

                if self.enable_metadata_cache and self.tensor_metadata_recv is None:
                    self.tensor_metadata_recv = create_send_metadata(input_tensor)

                return input_tensor, wait_handles
        return None, []

    def recv_backward(self, model_chunk_id: int, next_rank: int = None) -> Tuple[Any, List]:
        """Copy the gradient tensor from the next stage in pipeline as the input gradient of this stage.
           For interleaved 1F1B.

        Args:
            model_chunk_id (int): The current model chunk idx.
            next_rank (int, optional): The rank of the source of the tensor.

        Returns:
            Any: The input gradient tensor or gradient tensor list.
            Any: The wait handles for the communication.
        """
        with self.stage_manager.switch_model_chunk_id(model_chunk_id):
            if not self.stage_manager.is_last_stage():
                output_tensor_grad, wait_handles = self.comm.recv_backward(
                    next_rank, metadata_recv=self.grad_metadata_recv
                )
                if self.enable_metadata_cache and self.grad_metadata_recv is None:
                    self.grad_metadata_recv = create_send_metadata(output_tensor_grad)
                return output_tensor_grad, wait_handles

        return None, []

    def send_forward(self, model_chunk_id: int, output_tensor: Any, next_rank: int = None) -> List:
        """Sends the input tensor to the next stage in pipeline.
           For interleaved 1F1B.

        Args:
            model_chunk_id (int): The current model chunk idx.
            output_object (Any): Object to be sent.
            next_rank (int, optional): The rank of the recipient of the tensor.

        Returns:
            Any: The wait handles for the communication.
        """
        with self.stage_manager.switch_model_chunk_id(model_chunk_id):
            if not self.stage_manager.is_last_stage():
                send_handles = self.comm.send_forward(output_tensor, next_rank, send_metadata=self.send_tensor_metadata)
                self.send_tensor_metadata = not self.enable_metadata_cache
                return send_handles
        return []

    def send_backward(self, model_chunk_id: int, input_tensor_grad: Any, prev_rank: int = None) -> List:
        """Sends the gradient tensor to the previous stage in pipeline.
           For interleaved 1F1B.

        Args:
            model_chunk_id (int): The current model chunk idx.
            input_object (Any): Object to be sent.
            prev_rank (int, optional): The rank of the recipient of the tensor

        Returns:
            Any: The wait handles for the communication.
        """
        with self.stage_manager.switch_model_chunk_id(model_chunk_id):
            if not self.stage_manager.is_first_stage():
                send_handles = self.comm.send_backward(
                    input_tensor_grad, prev_rank, send_metadata=self.send_grad_metadata
                )
                self.send_grad_metadata = not self.enable_metadata_cache
                return send_handles
        return []

    def send_forward_recv_forward(
        self, model_chunk_id_send: int, model_chunk_id_recv: int, output_tensor: Any, send_first: bool = True
    ) -> Tuple[Any, List]:
        with self.stage_manager.switch_model_chunk_id(model_chunk_id_send):
            is_send = not self.stage_manager.is_last_stage()
        with self.stage_manager.switch_model_chunk_id(model_chunk_id_recv):
            is_recv = not self.stage_manager.is_first_stage()
        input_tensor, wait_handles = self.comm.send_forward_recv_forward(
            output_tensor,
            is_send,
            is_recv,
            send_metadata=self.send_tensor_metadata,
            metadata_recv=self.tensor_metadata_recv,
            send_first=send_first,
        )
        # Cache metadata
        self.send_tensor_metadata = not self.enable_metadata_cache and is_send
        if is_recv and self.enable_metadata_cache and self.tensor_metadata_recv is None:
            self.tensor_metadata_recv = create_send_metadata(input_tensor)

        return input_tensor, wait_handles

    def send_backward_recv_backward(
        self, model_chunk_id_send: int, model_chunk_id_recv: int, input_tensor_grad: Any, send_first: bool = True
    ) -> Tuple[Any, List]:
        with self.stage_manager.switch_model_chunk_id(model_chunk_id_send):
            is_send = not self.stage_manager.is_first_stage()
        with self.stage_manager.switch_model_chunk_id(model_chunk_id_recv):
            is_recv = not self.stage_manager.is_last_stage()
        output_tensor_grad, wait_handles = self.comm.send_backward_recv_backward(
            input_tensor_grad,
            is_send,
            is_recv,
            send_metadata=self.send_grad_metadata,
            metadata_recv=self.grad_metadata_recv,
            send_first=send_first,
        )
        # Cache metadata
        self.send_grad_metadata = not self.enable_metadata_cache and is_send
        if is_recv and self.enable_metadata_cache and self.grad_metadata_recv is None:
            self.grad_metadata_recv = create_send_metadata(output_tensor_grad)
        return output_tensor_grad, wait_handles

    def forward_step(
        self,
        model_chunk: Union[ModuleList, Module],
        model_chunk_id: int,
        input_obj: Optional[dict],
        criterion: Callable,
        accum_loss: Optional[torch.Tensor] = None,
        outputs: Optional[List[Any]] = None,
    ) -> Union[torch.Tensor, dict]:
        """Forward one step of the pipeline
        Args:
            model (ModuleList or Module): Model Chunk to be run
            input_obj (Optional[dict]): The output from the previous stage. If it is the first stage, the `input_obj` is None.
            criterion (Callable): Criterion to calculate loss.
            accum_loss (Optional[torch.Tensor], optional): Accumulated loss. Defaults to None.
            outputs (Optional[List[Any]], optional): List to store the output of the last stage (final output). Defaults to None.

        Returns:
            Union[torch.Tensor, dict]: The intermediate output (dict) of the current stage. If it is the last stage, the output is the loss (Tensor).
        """
        # Load input ids, attention mask and labels
        micro_batch = self.load_micro_batch(model_chunk_id=model_chunk_id)

        # for the first stage, input_obj is None
        # for other stages, input_obj is the output of the previous stage containing hidden_states etc.
        # Only attention_mask from micro_batch is used

        with self.stage_manager.switch_model_chunk_id(model_chunk_id):
            if isinstance(model_chunk, ModuleList):
                output_obj = model_forward(model_chunk[model_chunk_id], micro_batch, input_obj)
            else:
                # NOTE: in shardformer, each device still has the entire model, so we need to use relevant stage layers
                internal_inputs = {} if input_obj is None else input_obj
                internal_inputs["stage_index"] = self.stage_manager.stage_indices[model_chunk_id]
                output_obj = model_forward(model_chunk, micro_batch, internal_inputs)

            if self.stage_manager.is_last_stage():
                loss = criterion(output_obj, micro_batch) / self.num_microbatch
                if accum_loss is not None:
                    accum_loss.add_(loss.detach())
                if outputs is not None:
                    outputs.append(tree_map(detach, output_obj))
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

    def run_forward_only(
        self,
        model_chunk: Union[ModuleList, Module],
        data_iter: Iterable,
        criterion: Callable[..., Any],
        return_loss: bool = False,
        return_outputs: bool = False,
    ) -> Dict:
        assert self.forward_only

        self.load_batch(data_iter)

        outputs = [] if return_outputs and self.stage_manager.is_last_stage(ignore_chunk=True) else None

        accum_loss = None
        if return_loss and self.stage_manager.is_last_stage(ignore_chunk=True):
            accum_loss = torch.scalar_tensor(0, device=get_current_device())

        fwd_wait_handles = []
        model_chunk_id = self.get_model_chunk_id(0, is_forward=True)
        input_obj, fwd_wait_handles = self.recv_forward(model_chunk_id)

        for i in range(self.num_microbatch * self.num_model_chunks):
            last_batch = i == self.num_microbatch * self.num_model_chunks - 1
            model_chunk_id = self.get_model_chunk_id(i, is_forward=True)

            # Wait until current input is received
            _wait_p2p(fwd_wait_handles)
            output_obj = self.forward_step(model_chunk, model_chunk_id, input_obj, criterion, accum_loss, outputs)

            if not last_batch:
                input_obj, fwd_wait_handles = self.send_forward_recv_forward(
                    model_chunk_id_send=model_chunk_id,
                    model_chunk_id_recv=self.get_model_chunk_id(i + 1, is_forward=True),
                    output_tensor=output_obj,
                    send_first=self.stage_manager.stage % 2 == 0,
                )
            else:
                fwd_wait_handles = self.send_forward(model_chunk_id, output_obj)

        if outputs is not None:
            outputs = merge_batch(outputs)
        return {"loss": accum_loss, "outputs": outputs}

    def run_forward_backward(
        self,
        model_chunk: Union[ModuleList, Module],
        data_iter: Iterable,
        criterion: Callable[..., Any],
        optimizer: Optional[OptimizerWrapper] = None,
        return_loss: bool = False,
        return_outputs: bool = False,
    ) -> Dict:
        """
        Runs interleaved schedule, with communication between pipeline stages.
        """
        assert not self.forward_only

        self.load_batch(data_iter)

        num_microbatch = self.num_microbatch * self.num_model_chunks
        # Forward + until 1st backward
        num_warmup_microbatch = (self.stage_manager.num_stages - self.stage_manager.stage - 1) * 2
        # Steps needed to reach the last chunk
        num_warmup_microbatch += (self.num_model_chunks - 1) * self.stage_manager.num_stages
        num_warmup_microbatch = min(num_warmup_microbatch, num_microbatch)
        num_microbatch_remaining = num_microbatch - num_warmup_microbatch

        # Input, output tensors only need to be saved when doing backward passes
        input_objs = [[] for _ in range(self.num_model_chunks)]
        output_objs = [[] for _ in range(self.num_model_chunks)]

        outputs = [] if return_outputs and self.stage_manager.is_last_stage(ignore_chunk=True) else None

        accum_loss = None
        if return_loss and self.stage_manager.is_last_stage(ignore_chunk=True):
            accum_loss = torch.scalar_tensor(0, device=get_current_device())

        bwd_wait_handles = []
        # Get the 1st input batch
        model_chunk_id = self.get_model_chunk_id(0, is_forward=True)
        input_obj, fwd_wait_handles = self.recv_forward(model_chunk_id)

        # Run warmup forward passes.
        for i in range(num_warmup_microbatch):
            last_batch = i == num_warmup_microbatch - 1
            model_chunk_id = self.get_model_chunk_id(i, is_forward=True)

            # Wait for input
            _wait_p2p(fwd_wait_handles)
            output_obj = self.forward_step(model_chunk, model_chunk_id, input_obj, criterion, accum_loss, outputs)
            input_objs[model_chunk_id].append(input_obj)
            output_objs[model_chunk_id].append(output_obj)

            if last_batch and num_microbatch_remaining == 0:
                fwd_wait_handles = self.send_forward(model_chunk_id, output_obj)
            else:
                input_obj, fwd_wait_handles = self.send_forward_recv_forward(
                    model_chunk_id_send=model_chunk_id,
                    model_chunk_id_recv=self.get_model_chunk_id(i + 1, is_forward=True),
                    output_tensor=output_obj,
                    send_first=self.stage_manager.stage % 2 == 0,
                )

        if num_microbatch_remaining > 0:
            model_chunk_id = self.get_model_chunk_id(0, is_forward=False)
            output_obj_grad, bwd_wait_handles = self.recv_backward(model_chunk_id)

        # Run 1F1B in steady state.
        for i in range(num_microbatch_remaining):
            fwd_batch_id = i + num_warmup_microbatch
            last_batch = i == num_microbatch_remaining - 1
            model_chunk_id = self.get_model_chunk_id(fwd_batch_id, is_forward=True)

            # Wait for input.
            _wait_p2p(fwd_wait_handles)
            output_obj = self.forward_step(model_chunk, model_chunk_id, input_obj, criterion, accum_loss, outputs)
            # Add input_obj and output_obj to end of list.
            input_objs[model_chunk_id].append(input_obj)
            output_objs[model_chunk_id].append(output_obj)

            model_chunk_id = self.get_model_chunk_id(i, is_forward=False)
            # Pop output_obj and output_obj from the start of the list for the backward pass.
            _input_obj = input_objs[model_chunk_id].pop(0)
            _output_obj = output_objs[model_chunk_id].pop(0)

            # Helper functions
            def send_forward_recv_forward():
                if last_batch:
                    model_chunk_id = self.get_model_chunk_id(fwd_batch_id, is_forward=True)
                    wait_handles = self.send_forward(model_chunk_id, output_obj)
                    return None, wait_handles
                else:
                    input_obj, wait_handles = self.send_forward_recv_forward(
                        model_chunk_id_send=self.get_model_chunk_id(fwd_batch_id, is_forward=True),
                        model_chunk_id_recv=self.get_model_chunk_id(fwd_batch_id + 1, is_forward=True),
                        output_tensor=output_obj,
                        send_first=self.stage_manager.stage % 2 == 0
                        and i > 0,  # Receive from warmup stage first in the first batch
                    )
                    return input_obj, wait_handles

            def send_backward_recv_backward():
                no_cooldown = num_microbatch == num_microbatch_remaining
                if last_batch and no_cooldown:
                    model_chunk_id = self.get_model_chunk_id(i, is_forward=False)
                    wait_handles = self.send_backward(model_chunk_id, input_obj_grad)
                    return None, wait_handles
                else:
                    output_obj_grad, wait_handles = self.send_backward_recv_backward(
                        model_chunk_id_send=self.get_model_chunk_id(i, is_forward=False),
                        model_chunk_id_recv=self.get_model_chunk_id(i + 1, is_forward=False),
                        input_tensor_grad=input_obj_grad,
                        send_first=self.stage_manager.stage % 2 == 0,
                    )
                    return output_obj_grad, wait_handles

            input_obj, fwd_wait_handles = send_forward_recv_forward()
            # Wait for upstream grad
            _wait_p2p(bwd_wait_handles)
            input_obj_grad = self.backward_step(optimizer, _input_obj, _output_obj, output_obj_grad)
            # NOTE: It's documented by NCCL that running two concurrent communicators (batch_isend_irecv)
            # risks deadlock (https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2134/user-guide/docs/usage/communicators.html)
            # however in practice this works fine, and Megatron does this too
            # (https://github.com/microsoft/Megatron-DeepSpeed/blob/bcedecd1ff788d4d363f3365fd396053a08d65be/megatron/core/pipeline_parallel/schedules.py#L774)
            # if deadlock, call _wait_p2p(fwd_wait_handles) here
            output_obj_grad, bwd_wait_handles = send_backward_recv_backward()

        if num_microbatch_remaining == 0:
            model_chunk_id = self.get_model_chunk_id(0, is_forward=False)
            output_obj_grad, bwd_wait_handles = self.recv_backward(model_chunk_id)

        # Run cooldown backward passes.
        for i in range(num_microbatch_remaining, num_microbatch):
            last_batch = i == num_microbatch - 1
            model_chunk_id = self.get_model_chunk_id(i, is_forward=False)
            _input_obj = input_objs[model_chunk_id].pop(0)
            _output_obj = output_objs[model_chunk_id].pop(0)

            # Wait for upstream grad
            _wait_p2p(bwd_wait_handles)
            # backward local grads
            input_obj_grad = self.backward_step(optimizer, _input_obj, _output_obj, output_obj_grad)
            if not last_batch:
                output_obj_grad, bwd_wait_handles = self.send_backward_recv_backward(
                    model_chunk_id_send=self.get_model_chunk_id(i, is_forward=False),
                    model_chunk_id_recv=self.get_model_chunk_id(i + 1, is_forward=False),
                    input_tensor_grad=input_obj_grad,
                    send_first=self.stage_manager.stage % 2 == 0 and i > num_microbatch_remaining,
                )
                assert (not self.overlap_p2p) or len(bwd_wait_handles) > 0
            else:
                model_chunk_id = self.get_model_chunk_id(i, is_forward=False)
                _ = self.send_backward(model_chunk_id, input_obj_grad)

        assert all(len(v) == 0 for v in input_objs) and all(len(v) == 0 for v in output_objs)

        if outputs is not None:
            outputs = merge_batch(outputs)
        return {"loss": accum_loss, "outputs": outputs}

    def forward_backward_step(
        self,
        model_chunk: Union[ModuleList, Module],
        data_iter: Iterable,
        criterion: Callable[..., Any],
        optimizer: Optional[OptimizerWrapper] = None,
        return_loss: bool = False,
        return_outputs: bool = False,
    ) -> dict:
        """
        Args:
            model_chunk (ModuleList or Module): Model Chunk to be trained. Original interleaved uses a module list whereas shardformer uses entire model + layer specification
            data_iter (Iterable): Data iterator.
            criterion (Callable[[Any, Any], Tensor]): Criterion to be used. It should take two arguments: model outputs and inputs, and returns loss tensor.
            optimizer (OptimizerWrapper, optional): Optimizer to be used. Can be None when only forward is executed. Defaults to None.
            return_loss (bool, optional): Whether to return loss. Defaults to False. Whether to return loss.
            return_outputs (bool, optional): Whether to return model outputs. Defaults to False. Whether to return model outputs.

        Returns:
            dict: A dict with keys: 'loss' and 'outputs'.
        """
        self.forward_only = not torch.is_grad_enabled()
        if optimizer is None:
            assert self.forward_only, "Optimizer should be passed when doing backward."

        if self.forward_only:
            result = self.run_forward_only(model_chunk, data_iter, criterion, return_loss, return_outputs)
        else:
            result = self.run_forward_backward(
                model_chunk, data_iter, criterion, optimizer, return_loss, return_outputs
            )

        return result
