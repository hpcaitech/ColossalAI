from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import torch
import torch.cuda
import torch.distributed
from torch.nn import Module, ModuleList
from torch.utils._pytree import tree_flatten, tree_map

from colossalai.accelerator import get_accelerator
from colossalai.interface import OptimizerWrapper
from colossalai.pipeline.p2p import PipelineP2PCommunication, create_send_metadata
from colossalai.pipeline.schedule.v_schedule import ScheduledNode
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.pipeline.weight_grad_store import WeightGradStore

from ._utils import (
    clone,
    detach,
    get_batch_size,
    get_micro_batch,
    merge_batch,
    model_forward,
    release_tensor_data,
    require_grad,
    retain_grad,
    to_device,
)
from .base import PipelineSchedule

AUTO_SCHEDULE_COMMUNICATION_TYPES = {"RECV_FORWARD", "RECV_BACKWARD", "SEND_FORWARD", "SEND_BACKWARD"}


def _wait_p2p(wait_handles: List[torch.cuda.Event]) -> None:
    if wait_handles is not None:
        for req in wait_handles:
            req.wait()


class ZeroBubbleVPipeScheduler(PipelineSchedule):
    r"""
    ZeroBubbleVPipeScheduler

    Args:
        stage_manager (PipelineStageManager): If using pipeline parallelism, it's necessary to specify a pipeline stage manager for inter-process communication in pipeline parallelism. Defaults to None, which means not using pipeline parallelism.
        schedule (List[ScheduledNode]): Schedule for ZeroBubbleVPipe.
        num_model_chunks (int) : The number of model chunk in a device.
        num_microbatch (Optional[int]): The number of microbatch.
        microbatch_size (Optional[int]): The size per microbatch.
        enable_metadata_cache (bool): whether to enable metadata cache to acclerate communication.
        overlap_p2p (bool): whether to use overlap_p2p.
    """

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
        # batch info
        self.num_microbatch = num_microbatch
        self.microbatch_size = microbatch_size
        self.num_model_chunks = num_model_chunks
        self.batch: Any
        self.batch_size: int
        self.last_batch_size: Optional[int] = None
        self.microbatch_offset: List[int]

        self.schedules = schedule
        # TODO: optim post valid
        self.do_post_validation = False

        # P2PMeta cache
        self.enable_metadata_cache = enable_metadata_cache

        # check send_tensor_metadata, send_grad_metadata
        # pp4 as sample, we should follow this meta strategy
        #         send_tensor_meta(fwd)   send_grad_meta(bwd)
        #            chunk0 | chunk1        chunk0 | chunk 1
        # stage 0       T   |   F              F   |   T
        # stage 1       T   |   T              T   |   T
        # stage 2       T   |   T              T   |   T
        # stage 3       F   |   T              F   |   T
        if stage_manager.is_first_stage(ignore_chunk=True):
            self.send_tensor_metadata = [True, False]
            self.send_grad_metadata = [False, True]
        elif stage_manager.is_last_stage(ignore_chunk=True):
            self.send_tensor_metadata = [False, True]
            self.send_grad_metadata = [True, False]
        else:
            self.send_tensor_metadata = [True, True]
            self.send_grad_metadata = [True, True]

        # meta cache buffer
        self.tensor_metadata_recv = [None, None]  # [chunk 0 meta, chunk 1 meta]
        self.grad_metadata_recv = [None, None]

        # P2P communication
        self.comm = PipelineP2PCommunication(stage_manager, overlap_p2p=overlap_p2p)

        # init communication map
        self.communication_map = {
            "SEND_FORWARD": self.send_forward,
            "RECV_FORWARD": self.recv_forward,
            "SEND_BACKWARD": self.send_backward,
            "RECV_BACKWARD": self.recv_backward,
        }

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
        self.send_forward_buffer = [[], []]  # [chunk0:[torch.Tensor], chunk1:[torch.Tensor]]
        self.recv_forward_buffer = [
            [],
            [],
        ]  # [chunk0:[(torch.Tensor, wait_handle)], chunk1:[(torch.Tensor, wait_handle)]]
        self.send_backward_buffer = [[], []]  # [chunk0:[torch.Tensor], chunk1:[torch.Tensor]]
        self.recv_backward_buffer = [
            [],
            [],
        ]  # [chunk0:[(torch.Tensor, wait_handle)], chunk1:[(torch.Tensor, wait_handle)]]

        # y buffer for local send fwd
        self.local_send_forward_buffer = []
        # dy buffer for local send bwd
        self.local_send_backward_buffer = []

        # wait pp buffer
        self.wait_handles = []

    def assert_buffer_empty(self):
        # assert buffer is empty at end
        assert len(self.input_tensors[0]) == 0
        assert len(self.input_tensors[1]) == 0
        assert len(self.output_tensors[0]) == 0
        assert len(self.output_tensors[1]) == 0
        assert len(self.output_tensors_dw[0]) == 0
        assert len(self.output_tensors_dw[1]) == 0
        assert len(self.output_tensors_grad_dw[0]) == 0
        assert len(self.output_tensors_grad_dw[1]) == 0
        assert len(self.send_forward_buffer[0]) == 0
        assert len(self.send_forward_buffer[1]) == 0
        assert len(self.recv_forward_buffer[0]) == 0
        assert len(self.recv_forward_buffer[1]) == 0
        assert len(self.send_backward_buffer[0]) == 0
        assert len(self.send_backward_buffer[1]) == 0
        assert len(self.recv_backward_buffer[0]) == 0
        assert len(self.recv_backward_buffer[1]) == 0
        assert len(self.local_send_forward_buffer) == 0
        assert len(self.local_send_backward_buffer) == 0

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

    def recv_forward(self, model_chunk_id: int, prev_rank: int = None) -> List:
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
                    return []

                ################
                # chunk = 0 & not is_first_stage
                # Recv y from PREV_rank as input
                #################
                else:
                    prev_rank = self.stage_manager.get_prev_rank()
                    input_tensor, wait_handles = self.comm.recv_forward(
                        prev_rank=prev_rank, metadata_recv=self.tensor_metadata_recv[model_chunk_id]
                    )
                    if self.enable_metadata_cache and self.tensor_metadata_recv[model_chunk_id] is None:
                        self.tensor_metadata_recv[model_chunk_id] = create_send_metadata(input_tensor)
                    self.recv_forward_buffer[model_chunk_id].append((input_tensor, wait_handles))
                    return wait_handles

            else:
                ################
                # chunk = 1 & is_last_stage
                # do nothing; cause u get y from local_send_forward_buffer in schedule f
                ################
                if self.stage_manager.is_last_stage(ignore_chunk=True):
                    # return None, []
                    return []

                ################
                # chunk = 1 & not is_last_stage
                # recv y from NEXT_rank as input
                ################
                else:
                    next_rank = self.stage_manager.get_next_rank()
                    input_tensor, wait_handles = self.comm.recv_forward(
                        next_rank, metadata_recv=self.tensor_metadata_recv[model_chunk_id]
                    )
                    if self.enable_metadata_cache and self.tensor_metadata_recv[model_chunk_id] is None:
                        self.tensor_metadata_recv[model_chunk_id] = create_send_metadata(input_tensor)
                    self.recv_forward_buffer[model_chunk_id].append((input_tensor, wait_handles))
                    return wait_handles

    def recv_backward(self, model_chunk_id: int, next_rank: int = None) -> List:
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
                    return []

                ################
                # chunk = 0 & not is_last_stage
                # Recv bwd from next stage;
                ################
                else:
                    next_rank = self.stage_manager.get_next_rank()
                    output_tensor_grad, wait_handles = self.comm.recv_backward(
                        next_rank, metadata_recv=self.grad_metadata_recv[model_chunk_id]
                    )
                    if self.enable_metadata_cache and self.grad_metadata_recv[model_chunk_id] is None:
                        self.grad_metadata_recv[model_chunk_id] = create_send_metadata(output_tensor_grad)
                    self.recv_backward_buffer[model_chunk_id].append((output_tensor_grad, wait_handles))
                    return wait_handles

            else:
                # bwd chunk1 is left V;
                ################
                # chunk = 1 & is_first_stage
                # do nothing; get loss from local
                ################
                if self.stage_manager.is_first_stage(ignore_chunk=True):
                    return []

                ################
                # chunk = 1 & not first stage
                # recv_backward recv bwd from prev stage;
                ################
                else:
                    prev_rank = self.stage_manager.get_prev_rank()
                    output_tensor_grad, wait_handles = self.comm.recv_backward(
                        next_rank=prev_rank, metadata_recv=self.grad_metadata_recv[model_chunk_id]
                    )
                    if self.enable_metadata_cache and self.grad_metadata_recv[model_chunk_id] is None:
                        self.grad_metadata_recv[model_chunk_id] = create_send_metadata(output_tensor_grad)
                    self.recv_backward_buffer[model_chunk_id].append((output_tensor_grad, wait_handles))
                    return wait_handles

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
                    self.send_tensor_metadata[model_chunk_id] = not self.enable_metadata_cache
                    return []

                ################
                # chunk = 0 && not is_last_stage
                # self.comm.send_forward send y to NEXT stage
                ################
                else:
                    next_rank = self.stage_manager.get_next_rank()
                    output_tensor = self.send_forward_buffer[model_chunk_id].pop(0)
                    send_handles = self.comm.send_forward(
                        output_object=output_tensor,
                        next_rank=next_rank,
                        send_metadata=self.send_tensor_metadata[model_chunk_id],
                    )
                    self.send_tensor_metadata[model_chunk_id] = not self.enable_metadata_cache
                    return send_handles

            else:
                ################
                # chunk = 1 && is_first_stage
                # do nothing; Already send LOSS to local_send_backward_buffer in schedule f send part
                ################
                if self.stage_manager.is_first_stage(ignore_chunk=True):
                    self.send_tensor_metadata[model_chunk_id] = not self.enable_metadata_cache
                    return []

                ################
                # chunk = 1 && not is_first_stage
                # self.comm.send_forward send y to PREV stage
                ################
                else:
                    prev_rank = self.stage_manager.get_prev_rank()
                    output_tensor = self.send_forward_buffer[model_chunk_id].pop(0)
                    send_handles = self.comm.send_forward(
                        output_tensor, prev_rank, send_metadata=self.send_tensor_metadata[model_chunk_id]
                    )
                    self.send_tensor_metadata[model_chunk_id] = not self.enable_metadata_cache
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
                    self.send_grad_metadata[model_chunk_id] = not self.enable_metadata_cache
                    return []

                ################
                # chunk = 0 && not is_first_stage
                # Send dx to PREV stage;
                ################
                else:
                    prev_rank = self.stage_manager.get_prev_rank()
                    input_tensor_grad = self.send_backward_buffer[model_chunk_id].pop(0)
                    send_handles = self.comm.send_backward(
                        input_tensor_grad, prev_rank, send_metadata=self.send_grad_metadata[model_chunk_id]
                    )
                    self.send_grad_metadata[model_chunk_id] = not self.enable_metadata_cache
                    return send_handles

            # bwd chunk1 is left V;
            else:
                ################
                # chunk = 1 && is_last_stage
                # do nothing; Already send input_tensor_grad to local_send_bwd_buffer in schedule b;
                ################
                if self.stage_manager.is_last_stage(ignore_chunk=True):
                    self.send_grad_metadata[model_chunk_id] = not self.enable_metadata_cache
                    return []

                ################
                # chunk = 1 && not is_last_stage
                # Send dx to NEXT stage;
                ################
                else:
                    next_rank = self.stage_manager.get_next_rank()
                    input_tensor_grad = self.send_backward_buffer[model_chunk_id].pop(0)
                    send_handles = self.comm.send_backward(
                        input_tensor_grad, next_rank, send_metadata=self.send_grad_metadata[model_chunk_id]
                    )
                    self.send_grad_metadata[model_chunk_id] = not self.enable_metadata_cache
                    return send_handles

    def forward_step(
        self,
        model_chunk: Union[ModuleList, Module],
        model_chunk_id: int,
        micro_batch: Optional[dict],
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
        # for the first stage, input_obj is None; So,we use micro_batch as input_obj
        # for other stages, input_obj is the output of the previous/next stage containing hidden_states etc.
        # Only attention_mask from micro_batch is used
        with self.stage_manager.switch_model_chunk_id(model_chunk_id):
            #  fwd calculate
            internal_inputs = {} if input_obj is None else input_obj
            internal_inputs["stage_index"] = self.stage_manager.stage_indices[model_chunk_id]
            output_obj = model_forward(model_chunk, micro_batch, internal_inputs)
            # last layer in model
            if model_chunk_id == 1 and self.stage_manager.is_first_stage(ignore_chunk=True):
                loss = criterion(output_obj, micro_batch) / self.num_microbatch
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
        optimizer: OptimizerWrapper,
        # micro_batch: Optional[dict],
        input_obj: Optional[dict],
        output_obj: Union[dict, torch.Tensor],
        output_obj_grad: Optional[dict],
    ) -> Optional[dict]:
        """Backward dx step of the pipeline; we calculate "dx = w*dy" here;

        Args:
            model_chunk (ModuleList or Module): Model Chunk to be run;
            model_chunk_id (int): The current model chunk idx;
            optimizer (OptimizerWrapper): Optimizer to update the model
            input_obj (Optional[Tuple(dict)]): x. (microbatch, input_obj)
            output_obj (Union[dict, torch.Tensor]): y.
            output_obj_grad (dict): dy.

        Returns:
            Optional[dict]: dx.
        """
        # calculate bwd b step ; only dx = w*dy;

        # Retain the grad on the input_obj. No need retain_grad microbatch
        if input_obj is not None:
            tree_map(retain_grad, input_obj)

        # x, y, dy list for backward_by_grad; Type: list[tensor];
        input_obj_ = []
        output_obj_ = []
        output_obj_grad_ = []

        # For chunk 0 stage 0, use micro_batch as input_obj_; and we don't have to cal microbatch dx.

        # For loss backward; output_obj is loss; output_obj_grad should be None
        if model_chunk_id == 1 and self.stage_manager.is_first_stage(ignore_chunk=True):
            assert output_obj_grad is None
            input_obj_, _ = tree_flatten(input_obj)
            output_obj_.append(output_obj)  # LOSS
            output_obj_grad_.append(output_obj_grad)  # None

        # For other chunk stage, use input_obj as input_obj_;
        else:
            input_obj_, _ = tree_flatten(input_obj)
            output_obj_, _ = tree_flatten(output_obj)  # y
            output_obj_grad_, _ = tree_flatten(output_obj_grad)  # dy

        # filter item which is not torch.Tensor
        input_obj_ = [v for v in input_obj_ if isinstance(v, torch.Tensor) or v is None]
        output_obj_ = [v for v in output_obj_ if isinstance(v, torch.Tensor) or v is None]
        output_obj_grad_ = [v for v in output_obj_grad_ if isinstance(v, torch.Tensor) or v is None]

        try:
            ctx = optimizer.no_sync()
        except AttributeError:
            ctx = model_chunk.no_sync()
        with ctx:
            optimizer.backward_by_grad(
                tensor=output_obj_,
                grad=output_obj_grad_,
                # inputs=input_obj_,
                retain_graph=False,
            )
        # Format output_obj_grad
        input_obj_grad = dict()
        if model_chunk_id == 0 and self.stage_manager.is_first_stage(ignore_chunk=True):
            pass
        else:
            for k, v in input_obj.items():
                if isinstance(v, torch.Tensor) and v.grad is not None:
                    input_obj_grad[k] = v.grad
        return input_obj_grad

    def backward_w_step(
        self,
        model_chunk: Union[ModuleList, Module],
        model_chunk_id: int,
        optimizer: OptimizerWrapper,
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

        # y, dy list for w backward_by_grad; Type: list[tensor];
        output_obj_ = []
        output_obj_grad_ = []

        if model_chunk_id == 1 and self.stage_manager.is_first_stage(ignore_chunk=True):
            # loss backward; output_obj is loss;
            output_obj_.append(output_obj)  # LOSS
            output_obj_grad_.append(None)  # None
        else:
            output_obj_, _ = tree_flatten(output_obj)  # y
            output_obj_grad_, _ = tree_flatten(output_obj_grad)  # dy

        # filter item which is not torch.Tensor
        output_obj_ = [v for v in output_obj_ if isinstance(v, torch.Tensor) or v is None]
        output_obj_grad_ = [v for v in output_obj_grad_ if isinstance(v, torch.Tensor) or v is None]

        optimizer.backward_by_grad(
            tensor=output_obj_,
            grad=output_obj_grad_,
            inputs=list(model_chunk.parameters()),
            retain_graph=False,
        )

    def schedule_f(
        self,
        scheduled_node,
        model_chunk: torch.nn.ModuleList,
        model_chunk_id: int,
        criterion: Callable,
        accum_loss: Optional[torch.Tensor] = None,
        outputs: Optional[List[Any]] = None,
    ):
        """A complete forward schedule; Include recv fwd --> cal fwd --> send fwd;

        Args:
            scheduled_node:
            model_chunk (ModuleList or Module): Model Chunk to be run;
            model_chunk_id (int): The current model chunk idx;
            criterion (Callable): loss function;
            accum_loss (Optional[torch.Tensor], optional): Accumulated loss. Defaults to None.
            outputs (Optional[List[Any]], optional): List to store the output of the last stage (final output). Defaults to None.

        Returns:
            Nothing.
        """
        micro_batch = self.load_micro_batch(model_chunk_id=model_chunk_id)
        # Step1: recv fwd
        if model_chunk_id == 0:
            # is first stage; get input from microbatch
            if self.stage_manager.is_first_stage(ignore_chunk=True):
                input_obj = None  # (tensor, wait_handle)
            else:
                input_obj = self.recv_forward_buffer[model_chunk_id].pop(0)
                for h in input_obj[1]:
                    h.wait()
                input_obj = input_obj[0]
        else:
            # is last stage; recv from local
            if self.stage_manager.is_last_stage(ignore_chunk=True):
                input_obj = self.local_send_forward_buffer.pop(0)
            # not last stage; recv from next
            else:
                input_obj = self.recv_forward_buffer[model_chunk_id].pop(0)
                for h in input_obj[1]:
                    h.wait()
                input_obj = input_obj[0]
        # Here, let input_obj.requires_grad_()
        # if input_obj is not None:
        if not isinstance(input_obj, torch.Tensor):
            tree_map(require_grad, input_obj)

        # Also requires_grad_ for micro_batch in stage 0 chunk 0 fwd,
        # tree_map(torch.Tensor.requires_grad_, micro_batch)

        # Step2: fwd step
        output_obj = self.forward_step(
            model_chunk=model_chunk,
            model_chunk_id=model_chunk_id,
            micro_batch=micro_batch,
            input_obj=input_obj,
            criterion=criterion,
            accum_loss=accum_loss,
            outputs=outputs,
        )

        # Step3:
        # 3-1:detach output; detach output for send fwd;
        if model_chunk_id == 1 and self.stage_manager.is_first_stage(ignore_chunk=True):
            # We should not detach bwd LOSS
            pass
        else:
            # detach output
            detached_output_obj = tree_map(detach, output_obj)
            # 3-2 clone detached_output_obj
            detached_output_obj = tree_map(clone, detached_output_obj)

        # 3-3 release cloned output.data; release_tensor_data output for bwd b & w; (do not detach output)
        if model_chunk_id == 1 and self.stage_manager.is_first_stage(ignore_chunk=True):
            # We should not release_tensor_data bwd LOSS
            pass
        else:
            # release_tensor_data output
            tree_map(release_tensor_data, output_obj)

        # add input and output object for backward b
        self.input_tensors[model_chunk_id].append(input_obj)

        # for bwd b&w, we only need the graph(grad_fn) of output_obj
        # Do not release_tensor_data loss, release_tensor_data other output_obj;
        if model_chunk_id == 1 and self.stage_manager.is_first_stage(ignore_chunk=True):
            self.output_tensors[model_chunk_id].append(output_obj)
        else:
            self.output_tensors[model_chunk_id].append(output_obj)

        # add output to send_fwd_buffer
        if model_chunk_id == 0:  # chunk 0
            # is last stage; send to local_send_forward_buffer
            if self.stage_manager.is_last_stage(ignore_chunk=True):
                self.local_send_forward_buffer.append(detached_output_obj)
            else:
                self.send_forward_buffer[model_chunk_id].append(detached_output_obj)
        else:  # chunk 1
            # is first stage; end of fwd; do nothing
            if self.stage_manager.is_first_stage(ignore_chunk=True):
                pass
            else:
                self.send_forward_buffer[model_chunk_id].append(detached_output_obj)

    def schedule_b(
        self,
        scheduled_node,
        model_chunk: Union[ModuleList, Module],
        model_chunk_id: int,
        optimizer: OptimizerWrapper,
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
            # chunk0 not last stage; recv output_grad from recv_backward_buffer
            else:
                output_tensor_grad = self.recv_backward_buffer[model_chunk_id].pop(0)
                for h in output_tensor_grad[1]:
                    h.wait()
                output_tensor_grad = output_tensor_grad[0]
        else:
            # chunk1, is first stage; recv LOSS from local send bwd buffer
            if self.stage_manager.is_first_stage(ignore_chunk=True):
                output_tensor_grad = None
            # chunk1, not first stage; recv output_grad from recv_backward_buffer
            else:
                output_tensor_grad = self.recv_backward_buffer[model_chunk_id].pop(0)
                for h in output_tensor_grad[1]:
                    h.wait()
                output_tensor_grad = output_tensor_grad[0]

        # get input and output object from buffer;
        input_obj = self.input_tensors[model_chunk_id].pop(0)
        output_obj = self.output_tensors[model_chunk_id].pop(0)

        input_object_grad = self.backward_b_step(
            model_chunk=model_chunk,
            model_chunk_id=model_chunk_id,
            optimizer=optimizer,
            input_obj=input_obj,
            output_obj=output_obj,
            output_obj_grad=output_tensor_grad,
        )

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
        WeightGradStore.flush(chunk=model_chunk_id)

    def schedule_w(
        self,
        scheduled_node,
        model_chunk: Union[ModuleList, Module],
        model_chunk_id: int,
        optimizer: OptimizerWrapper,
    ):
        """A complete backward w schedule; Include get y & dy from buffer --> cal bwd w step(cal dw & update w);

        Args:
            scheduled_node:
            model_chunk (ModuleList or Module): Model Chunk to be run;
            model_chunk_id (int): The current model chunk idx;
        Returns:
            Nothing.
        """
        WeightGradStore.pop(chunk=model_chunk_id)

    def run_forward_only(
        self,
        model_chunk: Union[ModuleList, Module],
        data_iter: Iterable,
        criterion: Callable[..., Any],
        return_loss: bool = False,
        return_outputs: bool = False,
    ) -> Dict:
        assert self.forward_only

        # prepare batch
        self.load_batch(data_iter)

        # prepare accum loss & output
        accum_loss = None

        # reset accum loss at fwd end;
        if return_loss and self.stage_manager.is_first_stage(ignore_chunk=True):
            accum_loss = torch.scalar_tensor(0, device=get_accelerator().get_current_device())

        outputs = [] if return_outputs and self.stage_manager.is_first_stage(ignore_chunk=True) else None

        # while we still have schedules_node in self.schedules
        for it in range(len(self.schedules)):
            scheduled_node = self.schedules[it]

            if scheduled_node.type in {"RECV_FORWARD", "SEND_FORWARD"}:
                # communication
                communication_func = self.communication_map[scheduled_node.type]
                communication_func(scheduled_node.chunk)
            if scheduled_node.type == "F":
                self.schedule_f(
                    scheduled_node=scheduled_node,
                    model_chunk=model_chunk,
                    model_chunk_id=scheduled_node.chunk,
                    criterion=criterion,
                    accum_loss=accum_loss,
                    outputs=outputs,
                )
        # return loss & output
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
        Runs Zerobubble schedule, with communication between pipeline stages.
        """
        # prepare batch
        self.load_batch(data_iter)

        # prepare accum loss & output
        accum_loss = None

        # reset accum loss at fwd end;
        if return_loss and self.stage_manager.is_first_stage(ignore_chunk=True):
            accum_loss = torch.scalar_tensor(0, device=get_accelerator().get_current_device())

        outputs = [] if return_outputs and self.stage_manager.is_first_stage(ignore_chunk=True) else None

        # while we still have schedules_node in self.schedules
        schedule = self.schedules[self.stage_manager.stage]  # get schedule by stage (rank)
        for it in range(len(schedule)):
            scheduled_node = schedule[it]
            if scheduled_node.type in AUTO_SCHEDULE_COMMUNICATION_TYPES:
                # communication
                communication_func = self.communication_map[scheduled_node.type]
                wait_handle = communication_func(scheduled_node.chunk)
                # We wait recv handle in fwd step and bwd step. Here only need to wait for send handle
                if scheduled_node.type in {"SEND_FORWARD", "SEND_BACKWARD"}:
                    self.wait_handles.append(wait_handle)
            elif scheduled_node.type == "F":
                self.schedule_f(
                    scheduled_node=scheduled_node,
                    model_chunk=model_chunk,
                    model_chunk_id=scheduled_node.chunk,
                    criterion=criterion,
                    accum_loss=accum_loss,
                    outputs=outputs,
                )
            elif scheduled_node.type == "B":
                self.schedule_b(
                    scheduled_node=scheduled_node,
                    model_chunk=model_chunk,
                    model_chunk_id=scheduled_node.chunk,
                    optimizer=optimizer,
                )
            elif scheduled_node.type == "W":
                self.schedule_w(
                    scheduled_node=scheduled_node,
                    model_chunk=model_chunk,
                    model_chunk_id=scheduled_node.chunk,
                    optimizer=optimizer,
                )
        # wait here to ensure all communication is done
        for h in self.wait_handles:
            for hh in h:
                hh.wait()
        # return loss & output
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

        self.assert_buffer_empty()
        return result
