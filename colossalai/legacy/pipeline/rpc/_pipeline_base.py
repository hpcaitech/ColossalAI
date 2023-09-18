import inspect
import math
import threading
from abc import ABC, abstractmethod
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.distributed.rpc as rpc
from torch import autograd, nn, optim
from torch._C._distributed_rpc import PyRRef
from torch.futures import Future

from colossalai.legacy.pipeline.middleware import Partition, Topo
from colossalai.legacy.pipeline.pipeline_process_group import ppg
from colossalai.legacy.pipeline.rpc.utils import get_batch_lengths, pyobj_map, pytree_filter, pytree_map, split_batch


class Phase(Enum):
    FORWARD = 0
    BACKWARD = 1
    UPDATE = 2
    INPUT = 3


class UniqueKey:
    __slots__ = ("microbatch_id", "phase")
    microbatch_id: int
    phase: Phase

    def __init__(self, microbatch_id, phase) -> None:
        self.microbatch_id = microbatch_id
        self.phase = phase

    def __eq__(self, __o: object) -> bool:
        return (self.microbatch_id == __o.microbatch_id) and (self.phase == __o.phase)

    def __hash__(self) -> int:
        return tuple.__hash__((self.microbatch_id, self.phase))

    def __repr__(self) -> str:
        return f"Key(microbatch_id={self.microbatch_id}, phase={self.phase})"


class WorkItem:
    __slots__ = (
        "stage_id",
        "phase",
        "args",
        "kwargs",
        "output",
        "refcount",
        "microbatch_id",
        "batch_id",
        "num_microbatches",
        "forward_only",
    )

    stage_id: int
    phase: Phase
    args: Tuple[Any]
    kwargs: Dict[str, Any]
    output: Future
    microbatch_id: int
    refcount: int
    batch_id: int
    num_microbatches: int
    forward_only: bool

    def __init__(
        self, stage_id, phase, args, kwargs, output, microbatch_id, batch_id, num_microbatches, forward_only, refcount=0
    ) -> None:
        for attr_name in self.__slots__:
            setattr(self, attr_name, locals()[attr_name])


class BackwardCache:
    __slots__ = ("checkpoint", "stage_input_args", "stage_input_kwargs", "stage_outputs")
    checkpoint: bool
    stage_input_args: Tuple[Any]
    stage_input_kwargs: Dict[Any, Any]
    stage_outputs: Tuple[Any]

    def __init__(
        self,
        stage_input_args: Tuple[Any],
        stage_input_kwargs: Dict[Any, Any] = None,
        stage_outputs: Tuple[Any] = None,
        checkpoint: bool = False,
    ) -> None:
        for arg_name in self.__slots__:
            setattr(self, arg_name, locals()[arg_name])


class WorkerBase(ABC):
    def __init__(
        self,
        partition_fn: Callable,
        partition_args: tuple,
        pp_rank: int,
        actual_stage_num: int,
        num_microbatches: int,
        device: str,
        criterion: Callable = None,
        metric: Callable = None,
        checkpoint: bool = False,
        data_process_func: Callable = None,
    ) -> None:
        super().__init__()

        self.pp_rank = pp_rank
        self.actual_stage_num = actual_stage_num
        self.num_microbatches = num_microbatches
        self.checkpoint = checkpoint

        if data_process_func is not None:
            self.data_process_func = partial(data_process_func, pp_rank)

        self.device = device
        self._initialize_outstanding_range()

        # variable and const for context management
        self.outstanding = 0
        self.forward_times = 0
        self.backward_times = 0
        self.reset_key = UniqueKey(0, Phase.FORWARD)

        # rref of other workers
        self.pp_rank_to_worker_rref: Dict[int, PyRRef] = None

        # lock for the list
        self._initialize_lock()

        # topology info
        self.producer_stage_ids: List[int] = None
        self.consumer_stage_ids: List[int] = None

        # module partitions
        self.partition_fn = partition_fn
        self.partition_args = partition_args
        self.criterion = criterion
        self.metric = metric
        self.reset = False

        # context to maintain loop
        self._initialize_context_container()

        # main loop
        self.main_loop_thread = threading.Thread(target=self._work_loop, name=f"rank_{pp_rank}", daemon=True)
        self.main_loop_thread.start()

    def _get_future_by_device(self):
        return torch.futures.Future(devices=None if self.device in (None, "cpu") else [self.device])

    def _initialize_outstanding_range(self):
        outstanding_range = None
        if self.pp_rank == self.actual_stage_num - 1:
            outstanding_range = (0, 1)
        else:
            outstanding_range = (self.actual_stage_num, self.actual_stage_num)
        self.outstanding_range = outstanding_range

    def _initialize_context_container(self):
        self.microbatch_id_to_backward_cache: Dict[int, BackwardCache] = dict()
        self.microbatch_id_to_labels: Dict[int, Any] = dict()
        self.work_list: Dict[UniqueKey, WorkItem] = dict()
        self.output_list: Dict[UniqueKey, WorkItem] = dict()

    def _initialize_lock(self):
        self.partition_condition_lock = threading.Condition(threading.Lock())
        self.work_list_condition_lock = threading.Condition(threading.Lock())
        self.output_list_condition_lock = threading.Condition(threading.Lock())
        self.label_lock = threading.Condition(threading.Lock())
        self.reset_condition = threading.Condition(threading.Lock())

    def _initialize_partition(self):
        partition_fn = self.partition_fn
        partition_args = self.partition_args
        device = self.device
        with self.partition_condition_lock:
            self.module_partition: nn.Module = partition_fn(*partition_args).to(device)
            self.partition_condition_lock.notify_all()

    def _get_output_all(self, key: UniqueKey, ref_use=False, rank=None):
        with self.output_list_condition_lock:
            self.output_list_condition_lock.wait_for(lambda: key in self.output_list)
            output_work_item = self.output_list[key]
            output = output_work_item.output
            if not ref_use and output_work_item.phase != Phase.INPUT:
                self.output_list.pop(key)

        if not ref_use and output_work_item.phase != Phase.INPUT:
            output_work_item.refcount += 1
            refcount = output_work_item.refcount
            # lifecycle management for DAG scheduler
            if output_work_item.phase == Phase.FORWARD:
                lifecycle = len(self.get_consumer_stage_ids())
                if self.is_model_output():  # an extra reference for scheduler collecting results
                    lifecycle += 1
            elif output_work_item.phase == Phase.BACKWARD:
                lifecycle = len(self.get_producer_stage_ids())
                if self.is_model_input() and self._is_last_step(
                    output_work_item
                ):  # an extra reference for ensure_backward
                    lifecycle += 1
            else:
                lifecycle = 0
                refcount = 0

            with self.output_list_condition_lock:
                if refcount <= lifecycle:
                    self.output_list[key] = output_work_item
                    self.output_list_condition_lock.notify_all()

        if isinstance(output, Future):
            output = output.wait()

        return output

    def sync_global_worker_rrefs(self, pp_rank_to_worker_rref: Dict[int, PyRRef]) -> None:
        assert self.pp_rank_to_worker_rref is None, f"in rank {self.pp_rank}, worker has sync global workers rrefs"
        assert pp_rank_to_worker_rref is not None, "stage_to_workers must be a dict instead of None"
        self.pp_rank_to_worker_rref = pp_rank_to_worker_rref

        # for some schedule need the other worker's info to initialise partition (like Chimera)
        # construction of partition is executed after the registration of pp_rank_to_worker_rref
        self._initialize_partition()

    # res_use works for lifecycle counter,
    # if ref_use is True, lifecycle won't add.
    # offset supports get partial output to reduce comm costs.
    def get_output_by_key(self, key: UniqueKey, ref_use=False, rank=None, offsets=None) -> Any:
        output = self._get_output_all(key, ref_use, rank)
        if offsets is None:  # get all for non iterable output
            return output
        else:  # get part for iterable output
            output = [output[i] for i in offsets]
        return output

    def get_numels(self) -> int:
        numel = sum(param.numel() for param in self.module_partition.parameters())
        return numel

    def get_parameters(self) -> List[torch.Tensor]:
        return [p for p in self.module_partition.parameters()]

    def get_parameter_gradients(self) -> List[torch.Tensor]:
        return [p.grad for p in self.module_partition.parameters()]

    def get_partition(self):
        with self.partition_condition_lock:
            self.partition_condition_lock.wait_for(lambda: hasattr(self, "module_partition"))
            return self.module_partition

    def get_partition_state_dict(self):
        with self.partition_condition_lock:
            self.partition_condition_lock.wait_for(lambda: hasattr(self, "module_partition"))
            return self.module_partition.state_dict()

    def _make_args_kwargs(self, microbatch, merge=False):
        if isinstance(microbatch, dict):
            if merge:
                return list(microbatch.values()), {}
            return [], microbatch
        elif isinstance(microbatch, torch.Tensor):
            return [microbatch], {}
        elif isinstance(microbatch, (tuple, list)):
            args = []
            kwargs = {}
            for arg in microbatch:
                if isinstance(arg, dict):
                    kwargs.update(arg)
                else:
                    args.append(arg)
            if merge:
                arg_lst = args
                for arg in kwargs.values():
                    arg_lst.append(arg)
                return arg_lst, {}
            return args, kwargs
        else:
            raise TypeError(f"Input batch can be only dict, list, tuple or tensor, but receive {type(microbatch)}")

    # just for first pp_rank
    def set_input(self, microbatch_id: int, microbatch: Tuple[Any], forward_only: bool):
        key = UniqueKey(microbatch_id, Phase.FORWARD)
        output = self._get_future_by_device()

        if not self.use_middleware():
            # make args and kwargs
            args, kwargs = self._make_args_kwargs(microbatch)

            work_item = WorkItem(
                self.pp_rank,
                Phase.FORWARD,
                args,
                kwargs,
                output,
                microbatch_id,
                None,
                self.num_microbatches,
                forward_only,
            )
            with self.work_list_condition_lock:
                self.work_list[key] = work_item
                self.work_list_condition_lock.notify_all()
        else:
            # make args and kwargs
            arg_lst, _ = self._make_args_kwargs(microbatch, merge=True)

            # first stage assign correct input into other stages
            topo: Topo = self.get_topo()
            self_partition_id = self.pp_rank_to_partition_id(self.pp_rank, topo)
            input_partition = topo.get_input_partition()
            self_input_offsets = input_partition.get_output_offsets(self_partition_id)
            recv_input_key = UniqueKey(microbatch_id, Phase.INPUT)

            # set input for self rank
            self_arg_lst = []
            for off in self_input_offsets:
                self_arg_lst.append(arg_lst[off])

            work_item = WorkItem(
                self.pp_rank,
                Phase.FORWARD,
                self_arg_lst,
                {},
                output,
                microbatch_id,
                None,
                self.num_microbatches,
                forward_only,
            )
            with self.work_list_condition_lock:
                self.work_list[key] = work_item
                self.work_list_condition_lock.notify_all()

            # put input tensor which other nodes need into output_list as Phase.INPUT
            work_item_remote = WorkItem(
                self.pp_rank, Phase.INPUT, [], {}, arg_lst, microbatch_id, None, self.num_microbatches, forward_only
            )

            with self.output_list_condition_lock:
                self.output_list[recv_input_key] = work_item_remote
                self.output_list_condition_lock.notify_all()

    # just for last pp_rank
    def set_labels(self, microbatch_id: int, microlabels: Any):
        with self.label_lock:
            self.microbatch_id_to_labels[microbatch_id] = microlabels
            self.label_lock.notify_all()

    # just for last pp_rank
    def _begin_backward(self, microbatch_id: int):
        with self.work_list_condition_lock:
            assert self.producer_stage_ids is not None

            key = UniqueKey(microbatch_id, Phase.BACKWARD)
            output = self._get_future_by_device()
            grad_wrt_loss = None

            work_item = WorkItem(
                self.pp_rank,
                Phase.BACKWARD,
                grad_wrt_loss,
                {},
                output,
                microbatch_id,
                None,
                self.num_microbatches,
                False,
            )

            self.work_list[key] = work_item
            self.work_list_condition_lock.notify_all()

    def _subscribe_producer(self, microbatch_id: int, forward_only: bool):
        """
        You should call this function asynchronously
        """
        stage_id = self.pp_rank
        output = self._get_future_by_device()
        if not self.use_middleware():
            producer_num = len(self.producer_stage_ids)
            subscribe_forward_futures: List[Future] = [None] * producer_num
            for i in range(producer_num):
                producer_stage_id = self.producer_stage_ids[i]
                producer_output_key = UniqueKey(microbatch_id, Phase.FORWARD)
                producer_worker_rref = self.pp_rank_to_worker_rref[producer_stage_id]
                subscribe_forward_futures[i] = producer_worker_rref.rpc_async().get_output_by_key(producer_output_key)
        else:
            producer_stage_ids = self.get_producer_stage_ids()
            producer_num = len(producer_stage_ids)
            if self.need_model_input():
                producer_num += 1  # for input partition
            subscribe_forward_futures: List[Future] = [None] * producer_num

            # TODO(jiangziyue) get single value instead of the whole output
            if self.need_model_input():
                producer_stage_id = 0
                producer_output_key = UniqueKey(microbatch_id, Phase.INPUT)
                producer_worker_rref = self.pp_rank_to_worker_rref[producer_stage_id]
                offsets = self._get_input_offsets_by_index(target_index=0)
                subscribe_forward_futures[0] = producer_worker_rref.rpc_async().get_output_by_key(
                    producer_output_key, rank=self.pp_rank, offsets=offsets
                )

                for i in range(0, producer_num - 1):
                    producer_stage_id = producer_stage_ids[i]
                    producer_output_key = UniqueKey(microbatch_id, Phase.FORWARD)
                    producer_worker_rref = self.pp_rank_to_worker_rref[producer_stage_id]
                    target_index = i + 1
                    offsets = self._get_input_offsets_by_index(target_index=target_index)
                    if offsets is not None and len(offsets) == 0:  # no need to do rpc
                        subscribe_forward_futures[target_index] = []
                    else:
                        subscribe_forward_futures[target_index] = producer_worker_rref.rpc_async().get_output_by_key(
                            producer_output_key, rank=self.pp_rank, offsets=offsets
                        )

            else:
                for i in range(producer_num):
                    producer_stage_id = producer_stage_ids[i]
                    producer_output_key = UniqueKey(microbatch_id, Phase.FORWARD)
                    producer_worker_rref = self.pp_rank_to_worker_rref[producer_stage_id]
                    target_index = i
                    offsets = self._get_input_offsets_by_index(target_index=target_index)
                    if offsets is not None and len(offsets) == 0:  # no need to do rpc
                        subscribe_forward_futures[target_index] = []
                    else:
                        subscribe_forward_futures[target_index] = producer_worker_rref.rpc_async().get_output_by_key(
                            producer_output_key, rank=self.pp_rank, offsets=offsets
                        )

        work_item_from_producer = WorkItem(
            stage_id,
            Phase.FORWARD,
            subscribe_forward_futures,
            {},
            output,
            microbatch_id,
            None,
            self.num_microbatches,
            forward_only,
        )

        return work_item_from_producer

    # TODO(jiangziyue) Profile the side effect of the lock for lifecycle protection and consider a better one.
    def subscribe_producer(self, microbatch_id: int, forward_only: bool):
        key = UniqueKey(microbatch_id, Phase.FORWARD)
        with self.work_list_condition_lock:
            if key not in self.work_list:
                # On current PP middleware design for DAG, get_output_by_key used by _subscribe_producer
                # can only be executed once for every producer-consumer stage pair, which is necessary
                # to count the lifecycle of work_item. So, keeping the _subscribe_producer in the same
                # lock of work_item queue operation guarantees the consistency of lifecycle counter.
                work_item_from_producer = self._subscribe_producer(microbatch_id, forward_only)
                self.work_list[key] = work_item_from_producer
                self.work_list_condition_lock.notify_all()

    def _subscribe_consumer(self, microbatch_id: int):
        """
        You should call this function asynchronously
        """
        stage_id = self.pp_rank
        output = self._get_future_by_device()
        if not self.use_middleware():
            consumer_stage_ids = self.consumer_stage_ids
        else:
            consumer_stage_ids = self.get_consumer_stage_ids()
        consumer_num = len(consumer_stage_ids)
        subscribe_backward_futures: List[Future] = [None] * consumer_num
        for i in range(consumer_num):
            consumer_stage_id = consumer_stage_ids[i]
            consumer_output_key = UniqueKey(microbatch_id, Phase.BACKWARD)
            consumer_worker_rref = self.pp_rank_to_worker_rref[consumer_stage_id]
            target_index = i
            offsets = self._get_output_offsets_by_index(target_index=target_index)
            if offsets is not None and len(offsets) == 0:  # no need to do rpc
                subscribe_backward_futures[target_index] = []
            else:
                subscribe_backward_futures[target_index] = consumer_worker_rref.rpc_async().get_output_by_key(
                    consumer_output_key, rank=self.pp_rank, offsets=offsets
                )

        # flatten args
        work_item_from_consumer = WorkItem(
            stage_id,
            Phase.BACKWARD,
            subscribe_backward_futures,
            {},
            output,
            microbatch_id,
            None,
            self.num_microbatches,
            False,
        )

        return work_item_from_consumer

    def subscribe_consumer(self, microbatch_id: int):
        key = UniqueKey(microbatch_id, Phase.BACKWARD)
        with self.work_list_condition_lock:
            if key not in self.work_list:
                # On current PP middleware design for DAG, get_output_by_key used by subscribe_consumer
                # can only be executed once for every producer-consumer stage pair, which is necessary
                # to count the lifecycle of work_item. So, keeping the subscribe_consumer in the same
                # lock of work_item queue operation guarantees the consistency of lifecycle counter.
                work_item_from_consumer = self._subscribe_consumer(microbatch_id)
                self.work_list[key] = work_item_from_consumer
                self.work_list_condition_lock.notify_all()

    def get_producer_stage_ids(self):
        producer_stage_ids = []
        rank = self.pp_rank
        if not self.use_middleware():
            prev_rank = rank - 1
            if prev_rank >= 0:
                producer_stage_ids.append(prev_rank)
        else:
            topo: Topo = self.get_topo()
            self_partition_id = self.pp_rank_to_partition_id(rank, topo)
            self_partition: Partition = topo.get_partition_by_id(self_partition_id)
            input_partition_ids = self_partition.get_input_partition_ids()
            model_input_partition_id = topo.get_input_partition_id()
            for partition_id in input_partition_ids:
                # ignore input partition in current implementation.
                # it will be specially tackled.
                if partition_id != model_input_partition_id:
                    producer_stage_ids.append(self.partition_id_to_pp_rank(partition_id, topo))
        return producer_stage_ids

    def get_consumer_stage_ids(self):
        consumer_stage_ids = []
        rank = self.pp_rank
        if not self.use_middleware():
            next_rank = rank + 1
            if next_rank <= self.actual_stage_num - 1:
                consumer_stage_ids.append(next_rank)
        else:
            topo: Topo = self.get_topo()
            self_partition_id = self.pp_rank_to_partition_id(rank, topo)
            self_partition: Partition = topo.get_partition_by_id(self_partition_id)
            output_partition_ids = self_partition.get_output_partition_ids()
            model_output_partition_id = topo.get_output_partition_id()
            for partition_id in output_partition_ids:
                if model_output_partition_id != partition_id:
                    consumer_stage_ids.append(self.partition_id_to_pp_rank(partition_id, topo))
        return consumer_stage_ids

    def _get_producer_consumer(self) -> None:
        rank = self.pp_rank
        assert self.producer_stage_ids is None, f"all the producers of rank {rank} has been subscribed"
        assert self.consumer_stage_ids is None, f"all the consumers of rank {rank} has been subscribed"

        # should be arranged in order, the order of the input of current forward
        self.producer_stage_ids = self.get_producer_stage_ids()
        self.consumer_stage_ids = self.get_consumer_stage_ids()

    def pp_rank_to_partition_id(self, pp_rank: int, topo: Topo):
        partition_ids = topo.get_mid_partition_ids()
        return partition_ids[pp_rank]

    def partition_id_to_pp_rank(self, partition_id: int, topo: Topo):
        partition_ids = topo.get_mid_partition_ids()
        for i, id in enumerate(partition_ids):
            if id == partition_id:
                return i

    def get_topo(self):
        with self.partition_condition_lock:
            self.partition_condition_lock.wait_for(lambda: hasattr(self, "module_partition"))
            if hasattr(self.module_partition, "_topo"):
                return self.module_partition._topo
            else:
                return None

    def use_middleware(self):
        topo = self.get_topo()
        return topo is not None

    def _get_input_offsets_by_index(self, target_index):
        res = []
        topo: Topo = self.get_topo()
        self_partition_id = self.pp_rank_to_partition_id(self.pp_rank, topo)
        self_partition: Partition = topo.get_partition_by_id(self_partition_id)
        model_input_partition_id = topo.get_input_partition_id()
        input_vals = self_partition.get_input_vals()
        producer_stage_ids = self.get_producer_stage_ids()
        if self.need_model_input():
            # 0 for data from input batch
            # >= 1 for data from prev stages
            base = 1
        else:
            # data from prev stages
            base = 0
        for val in input_vals:
            val_pos = val.get()
            src_partition_id = val_pos.partition_id
            src_offset = val_pos.offset
            src_index = base
            src_partition = topo.get_partition_by_id(src_partition_id)
            output_len = len(src_partition.get_output_vals())
            # data from not-input partition
            if src_partition_id != model_input_partition_id:
                src_stage_id = self.partition_id_to_pp_rank(src_partition_id, topo)
                src_index = base
                for i, stage_id in enumerate(producer_stage_ids):
                    if stage_id == src_stage_id:
                        src_index += i
                        break
            else:  # data from input partition
                src_index = 0
            # when output_len = 1, not iterable
            if target_index == src_index:
                if output_len == 1:
                    res = None  # offset = None to get all outputs
                    return res
                else:
                    res.append(src_offset)
        return res

    def _get_output_offsets_by_index(self, target_index):
        res = []
        topo: Topo = self.get_topo()
        self_partition_id = self.pp_rank_to_partition_id(self.pp_rank, topo)
        self_partition: Partition = topo.get_partition_by_id(self_partition_id)
        output_vals = self_partition.get_output_vals()
        consumer_stage_ids = self.get_consumer_stage_ids()
        for val_list in output_vals:
            # An output may be passed to many down stages.
            for val_pos in val_list.get():
                dst_partition_id = val_pos.partition_id
                dst_offset = val_pos.offset
                dst_partition = topo.get_partition_by_id(dst_partition_id)
                input_len = len(dst_partition.get_input_vals())
                dst_stage_id = self.partition_id_to_pp_rank(dst_partition_id, topo)
                for i, stage_id in enumerate(consumer_stage_ids):
                    if stage_id == dst_stage_id:
                        dst_index = i
                        break
                if target_index == dst_index:
                    if input_len == 1:
                        res = None  # offset = None to get all outputs
                        return res
                    else:
                        res.append(dst_offset)
        return res

    # TODO(jiangziyue) get single value instead of the whole output
    def _get_real_args_kwargs_fwd(self, args_or_kwargs):
        if not self.use_middleware():
            args_or_kwargs = pytree_map(args_or_kwargs, fn=lambda x: x.wait(), process_types=Future)
            if args_or_kwargs is not None:
                if isinstance(args_or_kwargs, dict):
                    pass
                else:
                    flatten_args = []
                    pytree_map(args_or_kwargs, fn=lambda x: flatten_args.append(x), map_all=True)
                    args_or_kwargs = flatten_args
        else:
            args_or_kwargs = pytree_map(args_or_kwargs, fn=lambda x: x.wait(), process_types=Future)
            if args_or_kwargs is not None:
                if isinstance(args_or_kwargs, dict):
                    pass
                else:
                    flatten_args = []
                    if self.is_first_stage():
                        pytree_map(args_or_kwargs, fn=lambda x: flatten_args.append(x), map_all=True)
                    else:  # get by offset
                        topo: Topo = self.get_topo()
                        self_partition_id = self.pp_rank_to_partition_id(self.pp_rank, topo)
                        self_partition: Partition = topo.get_partition_by_id(self_partition_id)
                        model_input_partition_id = topo.get_input_partition_id()
                        input_vals = self_partition.get_input_vals()
                        producer_stage_ids = self.get_producer_stage_ids()
                        if self.need_model_input():
                            # 0 for data from input batch
                            # >= 1 for data from prev stages
                            base = 1
                        else:
                            # data from prev stages
                            base = 0
                        for val in input_vals:
                            val_pos = val.get()
                            src_partition_id = val_pos.partition_id
                            src_offset = val_pos.offset
                            src_index = base
                            src_partition = topo.get_partition_by_id(src_partition_id)
                            output_len = len(src_partition.get_output_vals())
                            # data from not-input partition
                            if src_partition_id != model_input_partition_id:
                                src_stage_id = self.partition_id_to_pp_rank(src_partition_id, topo)
                                src_index = base
                                for i, stage_id in enumerate(producer_stage_ids):
                                    if stage_id == src_stage_id:
                                        src_index += i
                                        break
                            else:  # data from input partition
                                src_index = 0
                            # when output_len = 1, not iterable
                            if output_len == 1:
                                target = args_or_kwargs[src_index]
                            else:
                                offsets = self._get_input_offsets_by_index(src_index)
                                real_offset = offsets.index(src_offset)
                                target = args_or_kwargs[src_index][real_offset]
                            flatten_args.append(target)
                    args_or_kwargs = flatten_args
        return args_or_kwargs

    # TODO(jiangziyue) get single value instead of the whole output
    def _get_real_args_kwargs_bwd(self, args_or_kwargs):
        if not self.use_middleware():
            args_or_kwargs = pytree_map(args_or_kwargs, fn=lambda x: x.wait(), process_types=Future)
            if args_or_kwargs is not None:
                if isinstance(args_or_kwargs, dict):
                    pass
                else:
                    flatten_args = []
                    pytree_map(args_or_kwargs, fn=lambda x: flatten_args.append(x), map_all=True)
                    args_or_kwargs = flatten_args
        else:
            for i, arg in enumerate(args_or_kwargs):
                args_or_kwargs[i] = arg.wait()
            if args_or_kwargs is not None:  # get by offset
                flatten_args = []
                topo: Topo = self.get_topo()
                self_partition_id = self.pp_rank_to_partition_id(self.pp_rank, topo)
                self_partition: Partition = topo.get_partition_by_id(self_partition_id)
                output_vals = self_partition.get_output_vals()
                consumer_stage_ids = self.get_consumer_stage_ids()
                for val_list in output_vals:
                    # An output may be passed to many down stages.
                    target = None
                    for val_pos in val_list.get():
                        dst_partition_id = val_pos.partition_id
                        dst_offset = val_pos.offset
                        dst_partition = topo.get_partition_by_id(dst_partition_id)
                        input_len = len(dst_partition.get_input_vals())
                        dst_stage_id = self.partition_id_to_pp_rank(dst_partition_id, topo)
                        for i, stage_id in enumerate(consumer_stage_ids):
                            if stage_id == dst_stage_id:
                                dst_index = i
                                break
                        if input_len == 1:
                            part_grad = args_or_kwargs[dst_index]
                        else:
                            offsets = self._get_output_offsets_by_index(dst_index)
                            real_offsets = offsets.index(dst_offset)
                            part_grad = args_or_kwargs[dst_index][real_offsets]

                        if target is None:
                            target = part_grad
                        elif part_grad is not None:
                            target += part_grad
                        else:
                            continue
                    flatten_args.append(target)
            args_or_kwargs = flatten_args
        return args_or_kwargs

    @abstractmethod
    def _get_work_item_key(self) -> UniqueKey:
        """
        this method control the order of the microbatch to consume
        """

    def is_first_stage(self):
        return self.pp_rank == 0

    def is_last_stage(self):
        return self.pp_rank == self.actual_stage_num - 1

    def need_model_input(self):
        need_input = False
        topo: Topo = self.get_topo()
        self_partition_id = self.pp_rank_to_partition_id(self.pp_rank, topo)
        self_partition = topo.get_partition_by_id(self_partition_id)
        partition_inputs = self_partition.get_input_partition_ids()
        model_input_partition_id = topo.get_input_partition_id()
        if model_input_partition_id in partition_inputs:
            need_input = True
        return not self.is_first_stage() and need_input

    def is_model_output(self):
        return self.is_last_stage()

    def is_model_input(self):
        return self.is_first_stage()

    def _default_data_process_func(self, args_kwargs):
        if self.is_first_stage():
            args = args_kwargs[0]
            kwargs = args_kwargs[1]
        else:
            args = args_kwargs
            kwargs = {}

        return args, kwargs

    def _consume_work_item_by_phase(self, work_item: WorkItem):
        phase = work_item.phase
        args = work_item.args
        kwargs = work_item.kwargs
        microbatch_id = work_item.microbatch_id
        forward_only = work_item.forward_only
        data_process_func = getattr(self, "data_process_func", self._default_data_process_func)
        consume_result = None

        is_first_stage = self.is_first_stage()
        is_last_stage = self.is_last_stage()

        if phase == Phase.FORWARD:
            # remind its consumer to get data before forward
            if not is_last_stage:
                for stage_id in self.consumer_stage_ids:
                    consumer_worker_rref = self.pp_rank_to_worker_rref[stage_id]
                    consumer_worker_rref.remote().subscribe_producer(microbatch_id, forward_only)

            # sustain pipeline context
            self.forward_times += 1
            if not forward_only:
                self.outstanding += 1

            # parse and integrate args and kwargs
            if is_first_stage:
                args = self._get_real_args_kwargs_fwd(args)
                kwargs = self._get_real_args_kwargs_fwd(kwargs)
                args_kwargs = (args, kwargs)
            else:
                args_kwargs = self._get_real_args_kwargs_fwd(args)

            args_kwargs = pyobj_map(
                args_kwargs, fn=lambda x: x.to(self.device).detach(), process_types=torch.Tensor
            )  # torch rpc doesn't support args or rets in GPU
            args_kwargs = pyobj_map(
                args_kwargs, fn=lambda x: self.device, process_types=torch.device
            )  # change devices from last stage to current device

            args, kwargs = data_process_func(args_kwargs)

            stage_outputs = None
            stage_input_args = args
            stage_input_kwargs = kwargs
            use_checkpoint = None

            if forward_only:
                with torch.no_grad():
                    consume_result = self.module_partition(*args, **kwargs)

                if is_last_stage and self.criterion:
                    with self.label_lock:
                        self.label_lock.wait_for(lambda: microbatch_id in self.microbatch_id_to_labels)
                    labels = self.microbatch_id_to_labels.pop(microbatch_id)
                    loss: torch.Tensor = self.criterion(consume_result, labels)
                    if self.metric is not None:
                        metric_result = self.metric(consume_result, labels)
                        if isinstance(metric_result, torch.Tensor):
                            metric_result = metric_result.item()
                    else:
                        metric_result = None
                    consume_result = [loss.item(), metric_result]

                # last stage doesn't need to do checkpoint, for it will do backward instantly
                stage_input_args = None
                stage_input_kwargs = None
                stage_outputs = consume_result

            elif self.checkpoint and not is_last_stage:
                with torch.no_grad():
                    consume_result = self.module_partition(*args, **kwargs)

                stage_outputs = consume_result
                use_checkpoint = True

            else:
                consume_result = self.module_partition(*args, **kwargs)

                if is_last_stage and self.criterion:
                    with self.label_lock:
                        self.label_lock.wait_for(lambda: microbatch_id in self.microbatch_id_to_labels)
                    labels = self.microbatch_id_to_labels.pop(microbatch_id)
                    loss: torch.Tensor = self.criterion(consume_result, labels)
                    if self.metric is not None:
                        metric_result = self.metric(consume_result, labels)
                        if isinstance(metric_result, torch.Tensor):
                            metric_result = metric_result.item()
                    else:
                        metric_result = None

                    consume_result = [loss.item(), metric_result]
                else:
                    loss = consume_result

                stage_outputs = loss
                use_checkpoint = False

            if not forward_only:
                self.microbatch_id_to_backward_cache[microbatch_id] = BackwardCache(
                    stage_input_args, stage_input_kwargs, stage_outputs, checkpoint=use_checkpoint
                )
            consume_result = pyobj_map(
                consume_result, fn=lambda x: x.to("cpu"), process_types=torch.Tensor
            )  # torch rpc doesn't support args or rets in

            # if not forward_only, do the backward
            if not forward_only:
                if is_last_stage:  # if it is the last stage, trigger backward automatic
                    self._begin_backward(microbatch_id)

        elif phase == Phase.BACKWARD:
            # remind its producer to get data before backward
            if not is_first_stage:
                for stage_id in self.producer_stage_ids:
                    producer_worker_rref = self.pp_rank_to_worker_rref[stage_id]
                    producer_worker_rref.remote().subscribe_consumer(microbatch_id)
            self.backward_times += 1
            self.outstanding -= 1

            assert (
                microbatch_id in self.microbatch_id_to_backward_cache
            ), f"microbatch_id {microbatch_id} not in backward cache"
            backward_cache = self.microbatch_id_to_backward_cache.pop(microbatch_id)

            stage_outputs = backward_cache.stage_outputs
            stage_input_args = backward_cache.stage_input_args
            stage_input_kwargs = backward_cache.stage_input_kwargs
            use_checkpoint = backward_cache.checkpoint

            if use_checkpoint:
                stage_outputs = [self.module_partition(*stage_input_args, **stage_input_kwargs)]

            # overlap recompute and future.wait
            if not is_last_stage:
                grad_tensors = self._get_real_args_kwargs_bwd(args)
            else:
                grad_tensors = None

            # take tensor only (for only tensor can do backward)
            # TODO(jiangziyue) : All values which should do bp are torch.Tensor?
            stage_outputs = pytree_filter(lambda x: True, stage_outputs, process_types=torch.Tensor)
            grad_tensors = pytree_filter(lambda x: True, grad_tensors, process_types=torch.Tensor)

            # output all input's grad to producer, even it has no grad(output None)
            # to make the offset aligned to the topo's record.
            if grad_tensors is not None:
                filtered_outputs = []
                filtered_grads = []
                for i, grad in enumerate(grad_tensors):
                    stage_output = stage_outputs[i]
                    if stage_output.requires_grad and grad is not None:
                        filtered_outputs.append(stage_output)
                        filtered_grads.append(grad)

                stage_outputs = filtered_outputs
                grad_tensors = pyobj_map(
                    filtered_grads, fn=lambda x: x.to(self.device), process_types=torch.Tensor
                )  # torch rpc doesn't support args or rets in GPU
            autograd.backward(stage_outputs, grad_tensors=grad_tensors)

            # collect grad of input tensor
            consume_result = []
            if not is_first_stage:
                # In current design, input mush be a flatten args.
                for arg in stage_input_args:
                    if isinstance(arg, torch.Tensor):
                        consume_result.append(arg.grad)
                    else:
                        consume_result.append(None)
                consume_result = pyobj_map(
                    consume_result, fn=lambda x: x.to("cpu"), process_types=torch.Tensor
                )  # torch rpc doesn't support args or rets in GPU

        else:
            raise TypeError(f"Unknown phase appears in _consume_work_item_by_phase {phase}")

        return consume_result

    def _get_store_len(self):
        return f"work_list:{len(self.work_list)} output_list:{len(self.output_list)} backward_cache:{len(self.microbatch_id_to_backward_cache)} label_cache:{len(self.microbatch_id_to_labels)}"

    def _get_parameter_grad_sum(self):
        grad_sum = 0
        for p in self.module_partition.parameters():
            if p.grad is not None:
                grad_sum += p.grad.sum()
        return grad_sum

    def _is_first_step(self, work_item: WorkItem) -> bool:
        return work_item.phase == Phase.FORWARD and work_item.microbatch_id == 0

    def _is_last_step(self, work_item: WorkItem) -> bool:
        if work_item.forward_only:
            last_phase = Phase.FORWARD
        else:
            last_phase = Phase.BACKWARD
        is_last_phase = work_item.phase == last_phase
        is_last_microbatch = work_item.microbatch_id == self.num_microbatches - 1
        return is_last_phase and is_last_microbatch

    def _hook_before_step(self):
        pass

    # install the main loop to wait for next batch input
    def _wait_for_reset(self):
        with self.reset_condition:
            self.reset_condition.wait_for(lambda: self.reset)
            self.reset = False

    # do the main loop to consume ready_list
    def _work_loop(self):
        # for init
        self._get_producer_consumer()
        torch.cuda.set_device(ppg.get_local_pp_rank())

        # main loop
        while True:
            work_item_key = self._get_work_item_key()
            # move current work item to output_list to activate subscribe in advance
            with self.work_list_condition_lock:
                self.work_list_condition_lock.wait_for(lambda: work_item_key in self.work_list)
                work_item = self.work_list[work_item_key]

            with self.output_list_condition_lock:
                # assert work_item_key not in self.output_list
                self.output_list[work_item_key] = work_item
                self.output_list_condition_lock.notify_all()

            consume_result = self._consume_work_item_by_phase(work_item)

            with self.work_list_condition_lock:
                self.work_list.pop(work_item_key)
            work_item.output.set_result(consume_result)

            # if is last step in one batch reset context and do step
            if self._is_last_step(work_item):
                self._wait_for_reset()

    # reset context and resume loop
    def reset_context(self):
        self.forward_times = 0
        self.backward_times = 0
        self.outstanding = 0
        self._initialize_outstanding_range()
        with self.work_list_condition_lock:
            self.work_list.clear()

        with self.output_list_condition_lock:
            self.output_list.clear()

        with self.reset_condition:
            self.reset = True
            self.reset_condition.notify_all()

    def initialize_optimizer(self, optimizer_class: type, **kwargs):
        self.optimizer: optim.Optimizer = optimizer_class(self.module_partition.parameters(), **kwargs)

    def step(self):
        self._hook_before_step()
        self.optimizer.step()
        self.optimizer.zero_grad()


class PipelineEngineBase(ABC, nn.Module):
    def __init__(
        self,
        worker_type,
        partition_fn: Callable,
        stage_num,
        num_microbatches,
        device: str,
        use_1F1B=False,
        chunk: int = 1,
        criterion: Callable = None,
        metric: Callable = None,
        checkpoint: bool = False,
        data_process_func: Callable = None,
    ) -> None:
        super().__init__()
        self.worker_type = worker_type
        self.partition_fn: Callable = partition_fn
        self.chunk = chunk
        self.criterion = criterion
        self.metric = metric
        self.num_microbatches = num_microbatches
        self.device = device
        self.use_1F1B = use_1F1B
        self.stage_num = stage_num
        self.checkpoint = checkpoint
        self.data_process_func = data_process_func

        self.pp_rank_to_worker_rref: Dict[int, PyRRef] = dict()

        self._check_argument()
        self._create_pp_rank_to_rpc_worker_id()
        self._create_pp_rank_to_module_partition_id()
        self._init_worker()

    def _check_argument(self) -> None:
        # make virtual stage num
        self.virtual_stage_num = self.stage_num * self.chunk
        assert self.stage_num <= torch.cuda.device_count(), "stage_num must be smaller than device count!"

        # check data_process_func
        data_process_func = self.data_process_func
        if data_process_func is not None:
            assert callable(data_process_func), "data_process_func must be a function"
            assert "<locals>" not in data_process_func.__repr__(), "data_process_func must be a global function"
            assert "<lambda>" not in data_process_func.__repr__(), "data_process_func cannot be a lambda expression"
            sig = inspect.signature(data_process_func)
            assert (
                len(sig.parameters) == 2
            ), f"length of data_process_func' arguments must be 2, receive {len(sig.parameters)} arguments instead"

    def _get_actual_stage_num(self) -> int:
        return self.stage_num if self.chunk == 1 else self.virtual_stage_num

    def _create_pp_rank_to_rpc_worker_id(self) -> None:
        """create a map from model partition to stage_id, which is useful when use_interleave is True.
        e.g. If a model is splited into 4 parts, which means stage_num is 2, chunk is 2, then
        pp_rank_to_rpc_worker_id = [0, 1, 0, 1], that means first and third part
        of partitions will be moved to device 0 and the others to device 1
        """
        stage_num = self.stage_num
        actual_stage_num = self._get_actual_stage_num()
        self.pp_rank_to_rpc_worker_id = [0] * actual_stage_num
        for pp_rank in range(actual_stage_num):
            self.pp_rank_to_rpc_worker_id[pp_rank] = pp_rank % stage_num

    def _create_pp_rank_to_module_partition_id(self) -> None:
        """By default(both fill drain and 1F1B), length of model partitions equal to
        actual_stage_num, so allocate model partition to corresponding stage
        """
        actual_stage_num = self._get_actual_stage_num()
        self.pp_rank_to_module_partition_id = [0] * actual_stage_num
        for pp_rank in range(actual_stage_num):
            self.pp_rank_to_module_partition_id[pp_rank] = pp_rank

    def _init_worker(self) -> None:
        actual_stage_num = self._get_actual_stage_num()

        worker_type = self.worker_type
        checkpoint = self.checkpoint
        num_microbatches = self.num_microbatches
        device = self.device
        criterion = self.criterion
        metric = self.metric
        partition_fn = self.partition_fn
        chunk = self.chunk
        data_process_func = self.data_process_func

        for pp_rank in range(len(self.pp_rank_to_rpc_worker_id)):
            partition_id = self.pp_rank_to_module_partition_id[pp_rank]
            partition_args = (partition_id, chunk, actual_stage_num)
            rpc_worker_id = self.pp_rank_to_rpc_worker_id[pp_rank]
            if device[:4] == "cuda":
                device = f"cuda:{rpc_worker_id}"
            self.pp_rank_to_worker_rref[pp_rank] = rpc.remote(
                rpc_worker_id,
                worker_type,
                args=(
                    partition_fn,
                    partition_args,
                    pp_rank,
                    actual_stage_num,
                    num_microbatches,
                    device,
                    criterion,
                    metric,
                    checkpoint,
                    data_process_func,
                ),
            )

        # let each worker know global worker rref (include itself)
        sync_futs = []
        for pp_rank in self.pp_rank_to_worker_rref:
            fut = (
                self.pp_rank_to_worker_rref[pp_rank]
                .rpc_async(timeout=0)
                .sync_global_worker_rrefs(self.pp_rank_to_worker_rref)
            )
            sync_futs.append(fut)

        for fut in sync_futs:
            fut.wait()

    def remote_numels(self) -> Dict[int, int]:
        numels = {}
        actual_stage_num = self._get_actual_stage_num()
        for stage_id in range(actual_stage_num):
            worker_rref = self.pp_rank_to_worker_rref[stage_id]
            numel = worker_rref.rpc_sync().get_numels()
            numels[stage_id] = numel
        return numels

    def remote_parameters(self) -> Dict[int, List[torch.Tensor]]:
        parameters = {}
        actual_stage_num = self._get_actual_stage_num()
        for stage_id in range(actual_stage_num):
            parameters[stage_id] = []
            worker_rref = self.pp_rank_to_worker_rref[stage_id]
            for p in worker_rref.rpc_sync().get_parameters():
                parameters[stage_id].append(p)
        return parameters

    def remote_grad(self) -> Dict[int, List[torch.Tensor]]:
        grads = {}
        actual_stage_num = self._get_actual_stage_num()
        for stage_id in range(actual_stage_num):
            grads[stage_id] = []
            worker_rref = self.pp_rank_to_worker_rref[stage_id]
            for grad in worker_rref.rpc_sync().get_parameter_gradients():
                grads[stage_id].append(grad)
        return grads

    def get_input_pp_ranks(self) -> List[int]:
        return [0]

    def get_output_pp_ranks(self) -> List[int]:
        return [self._get_actual_stage_num() - 1]

    def _consume_constraint(
        self, microbatch_id: int, forward_only: bool, input_pp_ranks: List[int], output_pp_ranks: List[int], ret_future
    ):
        actual_stage_num = self._get_actual_stage_num()
        use_1F1B = self.use_1F1B
        if microbatch_id >= actual_stage_num:
            if forward_only or not use_1F1B:
                for pp_rank in output_pp_ranks:
                    ret_future[pp_rank][microbatch_id - actual_stage_num].wait()
            else:
                key = UniqueKey(microbatch_id - actual_stage_num, Phase.BACKWARD)
                futs = []
                for pp_rank in input_pp_ranks:
                    worker_rref = self.pp_rank_to_worker_rref[pp_rank]
                    fut = worker_rref.rpc_async().get_output_by_key(key, ref_use=True, offsets=[])
                    futs.append(fut)

                for fut in futs:
                    fut.wait()

    def _create_ret_future(self, output_pp_ranks: List[int]) -> Dict[int, List[Future]]:
        num_microbatches = self.num_microbatches
        return {pp_rank: [None] * num_microbatches for pp_rank in output_pp_ranks}

    def _set_input(self, input_pp_ranks: List[int], microbatch_id: int, microbatch, forward_only: bool):
        for pp_rank in input_pp_ranks:
            worker_rref = self.pp_rank_to_worker_rref[pp_rank]
            # TODO : add relationship between input_pp_ranks and parts of microbatch
            worker_rref.remote().set_input(microbatch_id, microbatch, forward_only)

    def _set_labels(self, output_pp_ranks: List[int], microbatch_id: int, microlabels):
        for pp_rank in output_pp_ranks:
            worker_rref = self.pp_rank_to_worker_rref[pp_rank]
            # TODO : add relationship between output_pp_ranks and parts of microlabels
            worker_rref.remote().set_labels(microbatch_id, microlabels)

    # TODO(jiangziyue) : get model output with single value, instead of merging into last stage.
    def _subscribe_forward(self, microbatch_id: int, output_pp_ranks: List[int], ret_future: Dict[int, List[Future]]):
        key = UniqueKey(microbatch_id, Phase.FORWARD)
        for pp_rank in output_pp_ranks:
            worker_rref = self.pp_rank_to_worker_rref[pp_rank]
            ret_future[pp_rank][microbatch_id] = worker_rref.rpc_async().get_output_by_key(key)

    def _ensure_backward(self, forward_only: bool, input_pp_ranks: List[int]):
        if not forward_only:
            backward_result = []
            for pp_rank in input_pp_ranks:
                worker_rref = self.pp_rank_to_worker_rref[pp_rank]
                key = UniqueKey(self.num_microbatches - 1, Phase.BACKWARD)
                fut = worker_rref.rpc_async().get_output_by_key(
                    key, offsets=[]
                )  # only ensure the res exists, no need for real data.
                backward_result.append(fut)

            for fut in backward_result:
                fut.wait()

    def _collect_forward_result(self, output_pp_ranks: List[int], ret_future: Dict[int, List[Future]]):
        forward_result = []
        for pp_rank in output_pp_ranks:
            worker_forward_result = [None] * self.num_microbatches
            for microbatch_id in range(self.num_microbatches):
                ret = ret_future[pp_rank][microbatch_id].wait()
                # TODO : more stable format
                ret = [ret] if isinstance(ret, torch.Tensor) else ret
                worker_forward_result[microbatch_id] = ret

            worker_forward_result = list(zip(*worker_forward_result))
            forward_result.extend(worker_forward_result)

        return forward_result

    def _reset_worker(self):
        actual_stage_num = self._get_actual_stage_num()
        reset_futs: List[Future] = []
        for pp_rank in range(actual_stage_num):
            worker_rref = self.pp_rank_to_worker_rref[pp_rank]
            fut = worker_rref.rpc_async().reset_context()
            reset_futs.append(fut)

        for fut in reset_futs:
            fut.wait()

    def forward_backward(self, batch: torch.Tensor, labels: torch.Tensor = None, forward_only: bool = False):
        batch_lengths = get_batch_lengths(batch)
        batch_length = batch_lengths[0]

        if labels is not None and not forward_only:
            assert hasattr(
                self, "optimizer_class"
            ), "call `initialize_optimizer` to initialize optimizer before forward_backward"

        num_microbatches = self.num_microbatches

        assert (
            batch_length >= num_microbatches
        ), "num_microbatches is greater than the size of a batch, which is illegal"
        microbatch_size = math.ceil(batch_length / num_microbatches)
        device = self.device

        # If Chimera mode is used, then rank of down pipeline is excluded from 'input_pp_ranks' or 'output_pp_ranks'
        input_pp_ranks = self.get_input_pp_ranks()
        output_pp_ranks = self.get_output_pp_ranks()

        # a cache to collect data and control flow
        ret_future = self._create_ret_future(output_pp_ranks)

        for microbatch_id in range(num_microbatches):
            # control data input  speed
            # to prevent exceed of wait limitations
            # self._consume_constraint(microbatch_id, forward_only, input_pp_ranks, output_pp_ranks, ret_future)
            batch_start = microbatch_size * microbatch_id
            batch_end = min(batch_start + microbatch_size, batch_length)

            # set input
            microbatch = split_batch(batch, batch_start, batch_end, device)
            self._set_input(input_pp_ranks, microbatch_id, microbatch, forward_only)

            # set labels
            if labels is not None:
                # microlabels = labels[microbatch_size * microbatch_id:microbatch_size * (microbatch_id + 1)]
                microlabels = split_batch(labels, batch_start, batch_end, device)
                self._set_labels(output_pp_ranks, microbatch_id, microlabels)

            # get data asynchronously
            self._subscribe_forward(microbatch_id, output_pp_ranks, ret_future)

        # wait for first rank to ensure all backwards are done
        self._ensure_backward(forward_only, input_pp_ranks)

        # collect forward result
        forward_result = self._collect_forward_result(output_pp_ranks, ret_future)

        if not forward_only and hasattr(self, "optimizer_class"):
            self.step()

        self._reset_worker()  # reset worker attributes for next batch
        return forward_result

    def initialize_optimizer(self, optimizer_class: type, **kwargs):
        self.optimizer_class = optimizer_class
        for pp_rank in self.pp_rank_to_worker_rref:
            worker_rref = self.pp_rank_to_worker_rref[pp_rank]
            worker_rref.remote().initialize_optimizer(optimizer_class, **kwargs)

    def step(self):
        actual_stage_num = self._get_actual_stage_num()
        step_futs: List[Future] = []
        for pp_rank in range(actual_stage_num):
            worker_rref = self.pp_rank_to_worker_rref[pp_rank]
            fut = worker_rref.rpc_async().step()
            step_futs.append(fut)

        for fut in step_futs:
            fut.wait()
