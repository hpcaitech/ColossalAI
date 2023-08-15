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


class OneForwardOneBackwardSchedule(PipelineSchedule):

    def __init__(self, num_microbatches: int, stage_manager: PipelineStageManager) -> None:
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
        self.microbatch_offset = 0
        assert self.batch_size % self.num_microbatches == 0, \
            "Batch size should divided by the number of microbatches"
        self.microbatch_size = self.batch_size // self.num_microbatches

    def load_micro_batch(self) -> Any:
        """Load a micro batch from the current batch.

        Returns:
            Any: Micro batch.
        """
        micro_batch = get_micro_batch(self.batch, self.microbatch_offset, self.microbatch_size)
        self.microbatch_offset += self.microbatch_size
        return tree_map(partial(to_device, device=get_current_device()), micro_batch)

    def forward_step(self,
                     model: Module,
                     input_obj: Optional[dict],
                     criterion: Callable,
                     accum_loss: Optional[torch.Tensor] = None,
                     outputs: Optional[List[Any]] = None) -> Union[torch.Tensor, dict]:
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
                              model: Module,
                              optimizer: OptimizerWrapper,
                              data_iter: Iterable,
                              criterion: Callable[..., Any],
                              return_loss: bool = False,
                              return_outputs: bool = False) -> dict:
        """Runs non-interleaved 1F1B schedule, with communication between pipeline stages.

        Args:
            model (Module): Model to be trained.
            optimizer (OptimizerWrapper): Optimizer to be used.
            data_iter (Iterable): Data iterator.
            criterion (Callable[[Any, Any], Tensor]): Criterion to be used. It should take two arguments: model outputs and inputs, and returns loss tensor.
            return_loss (bool, optional): Whether to return loss. Defaults to False. Whether to return loss.
            return_outputs (bool, optional): Whether to return model outputs. Defaults to False. Whether to return model outputs.

        Returns:
            dict: A dict with keys: 'loss' and 'outputs'.
        """
        forward_only = not torch.is_grad_enabled()

        self.load_batch(data_iter)

        # num_warmup_microbatches is the step when not all the processes are working
        num_warmup_microbatches = self.stage_manager.num_stages - self.stage_manager.stage - 1
        num_warmup_microbatches = min(num_warmup_microbatches, self.num_microbatches)
        num_microbatches_remaining = self.num_microbatches - num_warmup_microbatches

        # Input, output tensors only need to be saved when doing backward passes
        input_objs = None
        output_objs = None

        if not forward_only:
            input_objs = []
            output_objs = []

        outputs = [] if return_outputs and self.stage_manager.is_last_stage() else None
        if return_loss and self.stage_manager.is_last_stage():
            accum_loss = torch.zeros(1, device=get_current_device())
        else:
            accum_loss = None

        # Run warmup forward passes.
        for i in range(num_warmup_microbatches):
            input_obj = self.comm.recv_forward()

            output_obj = self.forward_step(model, input_obj, criterion, accum_loss, outputs)

            self.comm.send_forward(output_obj)

            if not forward_only:
                input_objs.append(input_obj)
                output_objs.append(output_obj)

        # Before running 1F1B, need to receive first forward tensor.
        # If all microbatches are run in warmup / cooldown phase, then no need to
        # receive this tensor here.
        if num_microbatches_remaining > 0:
            input_obj = self.comm.recv_forward()

        # Run 1F1B in steady state.
        for i in range(num_microbatches_remaining):
            last_iteration = (i == (num_microbatches_remaining - 1))

            output_obj = self.forward_step(model, input_obj, criterion, accum_loss, outputs)
            if forward_only:
                self.comm.send_forward(output_obj)

                if not last_iteration:
                    input_obj = self.comm.recv_forward()

            else:
                # TODO adjust here
                self.comm.send_forward(output_obj)
                output_obj_grad = self.comm.recv_backward()

                # Add input_obj and output_obj to end of list.
                input_objs.append(input_obj)
                output_objs.append(output_obj)

                # Pop output_obj and output_obj from the start of the list for
                # the backward pass.
                input_obj = input_objs.pop(0)
                output_obj = output_objs.pop(0)
                input_obj_grad = self.backward_step(optimizer, input_obj, output_obj, output_obj_grad)

                if last_iteration:
                    input_obj = None
                else:
                    input_obj = self.comm.recv_forward()
                self.comm.send_backward(input_obj_grad)

        # Run cooldown backward passes.
        if not forward_only:
            for i in range(num_warmup_microbatches):
                input_obj = input_objs.pop(0)
                output_obj = output_objs.pop(0)

                output_obj_grad = self.comm.recv_backward()
                input_obj_grad = self.backward_step(optimizer, input_obj, output_obj, output_obj_grad)
                self.comm.send_backward(input_obj_grad)

        if outputs is not None:
            outputs = merge_batch(outputs)
        return {'loss': accum_loss, 'outputs': outputs}
