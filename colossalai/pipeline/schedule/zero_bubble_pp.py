from functools import partial
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

import torch
import torch.cuda
import torch.distributed
from torch.nn import Module, ModuleList
from torch.utils._pytree import tree_map

from colossalai.accelerator import get_accelerator
from colossalai.interface import OptimizerWrapper
from colossalai.pipeline.p2p import PipelineP2PCommunication
from colossalai.pipeline.schedule.v_schedule import ScheduledNode
from colossalai.pipeline.stage_manager import PipelineStageManager

from ._utils import detach, get_batch_size, get_micro_batch, retain_grad, to_device
from .base import PipelineSchedule

AUTO_SCHEDULE_COMMUNICATION_TYPES = {"RECV_FORWARD", "RECV_BACKWARD", "SEND_FORWARD", "SEND_BACKWARD"}


def _wait_p2p(wait_handles: List[torch.cuda.Event]) -> None:
    if wait_handles is not None:
        for req in wait_handles:
            req.wait()


class ZeroBubbleVPipeScheduler(PipelineSchedule):
    def __init__(
        self,
        stage_manager: PipelineStageManager,
        schedule: List[ScheduledNode],
        num_model_chunks: int,
        num_microbatch: Optional[int] = None,
        microbatch_size: Optional[int] = None,
        enable_metadata_cache: bool = True,
        overlap_p2p: bool = True,
    ):
        super().__init__(stage_manager)
        self.num_microbatch = num_microbatch
        self.collect_non_loss_data = None
        self.forward_only = None
        self.schedules = schedule
        # TODO: optim post valid
        self.do_post_validation = False
        self.is_first_run = True
        self.optimizer = None
        self.num_model_chunks = num_model_chunks

        # P2PMeta cache
        # self.enable_metadata_cache = enable_metadata_cache
        # self.send_tensor_metadata = True
        # self.send_grad_metadata = True
        # self.tensor_metadata_recv = None
        # self.grad_metadata_recv = None

        # P2P communication
        self.comm = PipelineP2PCommunication(stage_manager, overlap_p2p=overlap_p2p)

        # init buffer
        self._free_buffers()

    def _free_buffers(self):
        # free local buffer
        # two dim array, first dim is the model chunk, second dim is the microbatch queue

        # x & y buffer for schedule b
        self.input_tensors = [[], []]
        self.output_tensors = [[], []]

        # y & dy buffer for schedule w
        self.output_tensors_dw = [[], []]
        self.output_tensors_grad_dw = [[], []]

        # buffer for communication
        self.send_forward_buffer = [[], []]
        self.recv_forward_buffer = [[], []]
        self.send_backward_buffer = [[], []]
        self.recv_backward_buffer = [[], []]

        # y buffer for local send fwd
        self.local_send_forward_buffer = []
        # dy buffer for local send bwd
        self.local_send_backward_buffer = []

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
            # if self.batch_size != self.last_batch_size:
            # self.enable_metadata_cache = False
            # self.send_tensor_metadata = True
            # self.send_grad_metadata = True
            # self.tensor_metadata_recv = None
            # self.grad_metadata_recv = None

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
           For ZBV.

        Args:
            model_chunk_id (int): The current model chunk idx.
            prev_rank (int, optional): The rank of the source of the tensor.

        Returns:
            Any: The input tensor or input tensor list.
            Any: The wait handles for the communication.
        """
        with self.stage_manager.switch_model_chunk_id(model_chunk_id):
            if model_chunk_id == 0:
                ################
                # chunk = 0 & is_first_stage
                # do nothing; cause u are chunk 0 in first rank, u have no prev rank;
                #################
                if self.stage_manager.is_first_stage(ignore_chunk=True):
                    return None, []

                ################
                # chunk = 0 & not is_first_stage
                # Recv y from PREV_rank as input
                #################
                else:
                    prev_rank = self.stage_manager.get_prev_rank()
                    input_tensor, wait_handles = self.comm.recv_forward(prev_rank=prev_rank)
                    self.recv_forward_buffer[model_chunk_id].append(input_tensor)
                    return input_tensor, wait_handles

            else:
                ################
                # chunk = 1 & is_last_stage
                # do nothing; cause u get y from local_send_forward_buffer in schedule f
                ################
                if self.stage_manager.is_last_stage(ignore_chunk=True):
                    return None, []

                ################
                # chunk = 1 & not is_last_stage
                # recv y from NEXT_rank as input
                ################
                else:
                    next_rank = self.stage_manager.get_next_rank()
                    input_tensor, wait_handles = self.comm.recv_forward(next_rank)
                    self.recv_forward_buffer[model_chunk_id].append(input_tensor)
                    return input_tensor, wait_handles

    def recv_backward(self, model_chunk_id: int, next_rank: int = None) -> Tuple[Any, List]:
        """Copy the gradient tensor from the next stage in pipeline as the input gradient of this stage.
           For ZBV.

        Args:
            model_chunk_id (int): The current model chunk idx.
            next_rank (int, optional): The rank of the source of the tensor.

        Returns:
            Any: The input gradient tensor or gradient tensor list.
            Any: The wait handles for the communication.
        """
        with self.stage_manager.switch_model_chunk_id(model_chunk_id):
            if model_chunk_id == 0:
                # bwd chunk0 is right V;
                ################
                # chunk = 0 & is_last_stage
                # do nothing; Already get dy from local_send_backward_buffer in schedule b
                ################
                if self.stage_manager.is_last_stage(ignore_chunk=True):
                    return None, []

                ################
                # chunk = 0 & not is_last_stage
                # Recv bwd from next stage;
                ################
                else:
                    next_rank = self.stage_manager.get_next_rank()
                    output_tensor_grad, wait_handles = self.comm.recv_backward(next_rank)
                    self.recv_backward_buffer[model_chunk_id].append(output_tensor_grad)
                    return output_tensor_grad, wait_handles

            else:
                # bwd chunk1 is left V;
                ################
                # chunk = 1 & is_first_stage
                # do nothing; get loss from local
                ################
                if self.stage_manager.is_first_stage(ignore_chunk=True):
                    return None, []

                ################
                # chunk = 1 & not first stage
                # recv_backward recv bwd from prev stage;
                ################
                else:
                    prev_rank = self.stage_manager.get_prev_rank()
                    output_tensor_grad, wait_handles = self.comm.recv_backward(next_rank=prev_rank)
                    self.recv_backward_buffer[model_chunk_id].append(output_tensor_grad)
                    return output_tensor_grad, wait_handles

    def send_forward(self, model_chunk_id: int, next_rank: int = None) -> List:
        """Sends the input tensor to the next stage in pipeline.
           For ZBV.

        Args:
            model_chunk_id (int): The current model chunk idx.
            next_rank (int, optional): The rank of the recipient of the tensor.

        Returns:
            Any: The wait handles for the communication.
        """

        with self.stage_manager.switch_model_chunk_id(model_chunk_id):
            if model_chunk_id == 0:
                ################
                # chunk = 0 && is_last_stage
                # do nothing; hold y on local_send_forward_buffer
                ################
                if self.stage_manager.is_last_stage(ignore_chunk=True):
                    return []

                ################
                # chunk = 0 && not is_last_stage
                # self.comm.send_forward send y to NEXT stage
                ################
                else:
                    next_rank = self.stage_manager.get_next_rank()
                    output_tensor = self.send_forward_buffer[model_chunk_id].pop(0)
                    send_handles = self.comm.send_forward(output_object=output_tensor, next_rank=next_rank)
                    return send_handles

            else:
                ################
                # chunk = 1 && is_first_stage
                # do nothing; Already send LOSS to local_send_backward_buffer in schedule f send part
                ################
                if self.stage_manager.is_first_stage(ignore_chunk=True):
                    return []

                ################
                # chunk = 1 && not is_first_stage
                # self.comm.send_forward send y to PREV stage
                ################
                else:
                    prev_rank = self.stage_manager.get_prev_rank()
                    output_tensor = self.send_forward_buffer[model_chunk_id].pop(0)
                    send_handles = self.comm.send_forward(output_tensor, prev_rank)
                    return send_handles

    def send_backward(self, model_chunk_id: int, prev_rank: int = None) -> List:
        """Sends the gradient tensor to the previous stage in pipeline.
           For ZBV.

        Args:
            model_chunk_id (int): The current model chunk idx.
            prev_rank (int, optional): The rank of the recipient of the tensor

        Returns:
            Any: The wait handles for the communication.
        """

        with self.stage_manager.switch_model_chunk_id(model_chunk_id):
            if model_chunk_id == 0:
                # bwd chunk0 is right V;
                ################
                # chunk = 0 && is_first_stage
                # do nothing; cause u are the first chunk in first stage; bwd end
                ################
                if self.stage_manager.is_first_stage(ignore_chunk=True):
                    return []

                ################
                # chunk = 0 && not is_first_stage
                # Send dx to PREV stage;
                ################
                else:
                    prev_rank = self.stage_manager.get_prev_rank()
                    input_tensor_grad = self.send_backward_buffer[model_chunk_id].pop(0)
                    send_handles = self.comm.send_backward(input_tensor_grad, prev_rank)
                    return send_handles

            # bwd chunk1 is left V;
            else:
                # print(f"model_chunk_id {model_chunk_id} stage {self.stage_manager.stage} self.send_backward_buffer {self.send_backward_buffer}")
                ################
                # chunk = 1 && is_last_stage
                # do nothing; Already send input_tensor_grad to local_send_bwd_buffer in schedule b;
                ################
                if self.stage_manager.is_last_stage(ignore_chunk=True):
                    return []

                ################
                # chunk = 1 && not is_last_stage
                # Send dx to NEXT stage;
                ################
                else:
                    next_rank = self.stage_manager.get_next_rank()
                    input_tensor_grad = self.send_backward_buffer[model_chunk_id].pop(0)
                    send_handles = self.comm.send_backward(input_tensor_grad, next_rank)
                    return send_handles

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
            model_chunk (ModuleList or Module): Model Chunk to be run;
            model_chunk_id (int): The current model chunk idx;
            input_obj (Optional[dict]): x;
            criterion (Callable): loss function;
            accum_loss (Optional[torch.Tensor], optional): Accumulated loss. Defaults to None.
            outputs (Optional[List[Any]], optional): List to store the output of the last stage (final output). Defaults to None.

        Returns:
            Union[torch.Tensor, dict]: The intermediate output (dict) of the current stage. If it is the last stage, the output is the loss (Tensor).
        """
        # Load input ids, attention mask and labels
        # micro_batch = self.load_micro_batch(model_chunk_id=model_chunk_id)

        # for the first stage, input_obj is None
        # for other stages, input_obj is the output of the previous/next stage containing hidden_states etc.
        # Only attention_mask from micro_batch is used

        with self.stage_manager.switch_model_chunk_id(model_chunk_id):
            # fwd calculate
            output_obj = model_chunk[model_chunk_id](input_obj)
            # last layer in model
            if model_chunk_id == 1 and self.stage_manager.is_first_stage(ignore_chunk=True):
                loss = criterion(output_obj) / self.num_microbatch
                if accum_loss is not None:
                    accum_loss.add_(loss.detach())
                if outputs is not None:
                    outputs.append(tree_map(detach, output_obj))
                return loss
            else:
                return output_obj

    def backward_b_step(
        self,
        model_chunk: Union[ModuleList, Module],
        model_chunk_id: int,
        # optimizer: OptimizerWrapper,
        input_obj: Optional[dict],
        output_obj: Union[dict, torch.Tensor],
        output_obj_grad: Optional[dict],
    ) -> Optional[dict]:
        """Backward dx step of the pipeline; we calculate "dx = w*dy" here;

        Args:
            model_chunk (ModuleList or Module): Model Chunk to be run;
            model_chunk_id (int): The current model chunk idx;
            optimizer (OptimizerWrapper): Optimizer to update the model
            input_obj (Optional[dict]): x.
            output_obj (Union[dict, torch.Tensor]): y.
            output_obj_grad (dict): dy.

        Returns:
            Optional[dict]: dx.
        """
        # calculate bwd b step ; only dx = w*dy;

        # Retain the grad on the input_obj.
        tree_map(retain_grad, input_obj)

        if model_chunk_id == 0:
            # bwd step
            torch.autograd.backward(
                tensors=output_obj, grad_tensors=output_obj_grad, inputs=input_obj, retain_graph=True
            )
        else:
            if self.stage_manager.is_first_stage(ignore_chunk=True):
                # loss backward; output_obj is loss
                torch.autograd.backward(output_obj, inputs=input_obj, retain_graph=True)
            else:
                # commom bwd step
                # BUG:output_obj_grad is None
                torch.autograd.backward(
                    tensors=output_obj, grad_tensors=output_obj_grad, inputs=input_obj, retain_graph=True
                )

        return input_obj.grad

    def backward_w_step(
        self,
        model_chunk: Union[ModuleList, Module],
        model_chunk_id: int,
        # optimizer: OptimizerWrapper,
        output_obj: Union[dict, torch.Tensor],
        output_obj_grad: Optional[dict],
    ):
        """Backward dw step of the pipeline; we calculate "dw = x*dy" here;

        Args:
            model_chunk (ModuleList or Module): Model Chunk to be run;
            model_chunk_id (int): The current model chunk idx;
            optimizer (OptimizerWrapper): Optimizer to update the model
            output_obj (Union[dict, torch.Tensor]): y.
            output_obj_grad (dict): dy.

        Returns:
            Nothing need to return; we only calculate dw then update w;
        """
        # calculate bwd w step ; only dw = x*dy;
        if model_chunk_id == 0:
            torch.autograd.backward(
                tensors=output_obj, grad_tensors=output_obj_grad, inputs=list(model_chunk[model_chunk_id].parameters())
            )

        else:
            if self.stage_manager.is_first_stage(ignore_chunk=True):
                torch.autograd.backward(output_obj_grad, inputs=list(model_chunk[model_chunk_id].parameters()))
            else:
                torch.autograd.backward(
                    tensors=output_obj,
                    grad_tensors=output_obj_grad,
                    inputs=list(model_chunk[model_chunk_id].parameters()),
                )

    def schedule_f(
        self,
        scheduled_node,
        model_chunk: torch.nn.ModuleList,
        model_chunk_id: int,
        input_obj: Optional[dict],
        criterion: Callable,
        accum_loss: Optional[torch.Tensor] = None,
        outputs: Optional[List[Any]] = None,
    ):
        """A complete forward schedule; Include recv fwd --> cal fwd --> send fwd;

        Args:
            scheduled_node:
            model_chunk (ModuleList or Module): Model Chunk to be run;
            model_chunk_id (int): The current model chunk idx;
            input_obj (Optional[dict]): x;
            criterion (Callable): loss function;
            accum_loss (Optional[torch.Tensor], optional): Accumulated loss. Defaults to None.
            outputs (Optional[List[Any]], optional): List to store the output of the last stage (final output). Defaults to None.

        Returns:
            Nothing.
        """
        # Step1: recv fwd
        if model_chunk_id == 0:
            # is first stage; get input from func param
            if self.stage_manager.is_first_stage(ignore_chunk=True):
                input_obj = input_obj
            else:
                input_obj = self.recv_forward_buffer[model_chunk_id].pop(0)
        else:
            # is last stage; recv from local
            if self.stage_manager.is_last_stage(ignore_chunk=True):
                input_obj = self.local_send_forward_buffer.pop(0)
            # not last stage; recv from next
            else:
                input_obj = self.recv_forward_buffer[model_chunk_id].pop(0)

        # Step2: fwd step
        output_obj = self.forward_step(
            model_chunk=model_chunk,
            model_chunk_id=model_chunk_id,
            input_obj=input_obj,
            criterion=criterion,
            accum_loss=accum_loss,
            outputs=outputs,
        )

        # add input and output object for backward b
        self.input_tensors[model_chunk_id].append(input_obj)
        self.output_tensors[model_chunk_id].append(output_obj)

        # add output object for backward w
        self.output_tensors_dw[model_chunk_id].append(output_obj)

        # Step3: send fwd
        # add output to send_fwd_buffer
        if model_chunk_id == 0:
            # is last stage; send to local_send_forward_buffer
            if self.stage_manager.is_last_stage(ignore_chunk=True):
                self.local_send_forward_buffer.append(output_obj)
            else:
                self.send_forward_buffer[model_chunk_id].append(output_obj)
        else:
            # is first stage; end of fwd; append LOSS to local_send_backward_buffer
            if self.stage_manager.is_first_stage(ignore_chunk=True):
                self.local_send_backward_buffer.append(output_obj)
            else:
                self.send_forward_buffer[model_chunk_id].append(output_obj)

    def schedule_b(
        self,
        scheduled_node,
        model_chunk: Union[ModuleList, Module],
        model_chunk_id: int,
        # optimizer: OptimizerWrapper,
        # input_obj: Optional[dict],
        # output_obj: Union[dict, torch.Tensor],
        # output_obj_grad: Optional[dict],
    ):
        """A complete backward b schedule; Include recv bwd --> cal bwd step --> send bwd;

        Args:
            scheduled_node:
            model_chunk (ModuleList or Module): Model Chunk to be run;
            model_chunk_id (int): The current model chunk idx;
        Returns:
            Nothing.
        """

        # Step1: recv bwd
        if model_chunk_id == 0:
            # chunk0 is last stage; recv output_grad from local_send_backward_buffer
            if self.stage_manager.is_last_stage(ignore_chunk=True):
                output_tensor_grad = self.local_send_backward_buffer.pop(0)
            # chunk 0 not last stage; recv output_grad from recv_backward_buffer
            else:
                output_tensor_grad = self.recv_backward_buffer[model_chunk_id].pop(0)
        else:
            # chunk1, is first stage; recv LOSS from local send bwd buffer
            if self.stage_manager.is_first_stage(ignore_chunk=True):
                output_tensor_grad = self.local_send_backward_buffer.pop(0)
            # chunk1, not first stage; recv output_grad  from recv_backward_buffer
            else:
                output_tensor_grad = self.recv_backward_buffer[model_chunk_id].pop(0)

        # print(f"model_chunk_id {model_chunk_id} stage {self.stage_manager.stage}; output_tensor_grad {output_tensor_grad}\n")

        # get input and output object from buffer;
        input_obj = self.input_tensors[model_chunk_id].pop(0)
        output_obj = self.output_tensors[model_chunk_id].pop(0)

        # save output_tensor_grad for dw
        if model_chunk_id == 1 and self.stage_manager.is_first_stage(ignore_chunk=True):
            # we save loss here
            self.output_tensors_grad_dw[model_chunk_id].append(output_obj)
        else:
            # we save output_tensor_grad here
            self.output_tensors_grad_dw[model_chunk_id].append(output_tensor_grad)

        # _wait_p2p(recv_bwd_handles)
        # Step2: bwd step
        input_object_grad = self.backward_b_step(
            model_chunk=model_chunk,
            model_chunk_id=model_chunk_id,
            # optimizer: OptimizerWrapper,
            input_obj=input_obj,
            output_obj=output_obj,
            output_obj_grad=output_tensor_grad,
        )
        # print(f"model_chunk_id {model_chunk_id}; stage {self.stage_manager.stage}; input_object_grad {input_object_grad}")

        # Step3: send bwd
        if model_chunk_id == 0:
            # do nothing; end of bwd;
            if self.stage_manager.is_first_stage(ignore_chunk=True):
                pass
            # save input_object_grad to send_backward_buffer
            else:
                self.send_backward_buffer[model_chunk_id].append(input_object_grad)
        else:
            # send to local_send_backward_buffer
            if self.stage_manager.is_last_stage(ignore_chunk=True):
                self.local_send_backward_buffer.append(input_object_grad)
            # send to next
            else:
                self.send_backward_buffer[model_chunk_id].append(input_object_grad)

    def schedule_w(
        self,
        scheduled_node,
        model_chunk: Union[ModuleList, Module],
        model_chunk_id: int,
        # optimizer: OptimizerWrapper,
    ):
        """A complete backward w schedule; Include get y & dy from buffer --> cal bwd w step(cal dw & update w);

        Args:
            scheduled_node:
            model_chunk (ModuleList or Module): Model Chunk to be run;
            model_chunk_id (int): The current model chunk idx;
        Returns:
            Nothing.
        """

        # get y & dy from buffer
        output_obj = self.output_tensors_dw[model_chunk_id].pop(0)
        output_obj_grad = self.output_tensors_grad_dw[model_chunk_id].pop(0)

        self.backward_w_step(
            model_chunk=model_chunk,
            model_chunk_id=model_chunk_id,
            # optimizer: OptimizerWrapper,
            output_obj=output_obj,
            output_obj_grad=output_obj_grad,
        )

    def run_forward_backward(
        self,
        model_chunk: Union[ModuleList, Module],
        input_obj: Optional[dict],
        data_iter: Iterable,
        criterion: Callable[..., Any],
        optimizer: Optional[OptimizerWrapper] = None,
        return_loss: bool = False,
        return_outputs: bool = False,
    ):
        """
        Runs Zerobubble schedule, with communication between pipeline stages.
        """
        it = 0
        # while we still have schedules_node in self.schedules
        while it < len(self.schedules):
            scheduled_node = self.schedules[it]
            print(
                f"it {it}; manger_stage {self.stage_manager.stage}; node_stage {scheduled_node.stage} chunk {scheduled_node.chunk} {scheduled_node.type};"
            )
            if scheduled_node.type in AUTO_SCHEDULE_COMMUNICATION_TYPES:
                # communication
                if scheduled_node.type == "RECV_FORWARD":
                    self.recv_forward(scheduled_node.chunk)
                elif scheduled_node.type == "RECV_BACKWARD":
                    self.recv_backward(scheduled_node.chunk)
                elif scheduled_node.type == "SEND_FORWARD":
                    self.send_forward(scheduled_node.chunk)
                elif scheduled_node.type == "SEND_BACKWARD":
                    self.send_backward(scheduled_node.chunk)
            if scheduled_node.type == "F":
                self.schedule_f(
                    scheduled_node=scheduled_node,
                    model_chunk=model_chunk,
                    model_chunk_id=scheduled_node.chunk,
                    input_obj=input_obj,
                    criterion=criterion,
                    accum_loss=return_loss,
                    outputs=return_outputs,
                )
            elif scheduled_node.type == "B":
                self.schedule_b(
                    scheduled_node=scheduled_node,
                    model_chunk=model_chunk,
                    model_chunk_id=scheduled_node.chunk,
                )
            elif scheduled_node.type == "W":
                self.schedule_w(
                    scheduled_node=scheduled_node,
                    model_chunk=model_chunk,
                    model_chunk_id=scheduled_node.chunk,
                )
            it += 1
