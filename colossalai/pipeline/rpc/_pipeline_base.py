import threading
from enum import Enum
from typing import List, Any, Tuple, Dict, Callable
from functools import partial
from abc import ABC, abstractmethod
import sys
import os
import inspect

import torch
from torch import nn
import torch.distributed.rpc as rpc
from torch.futures import Future
from torch._C._distributed_rpc import PyRRef
from torch import autograd
from torch import optim

from colossalai.pipeline.pipeline_process_group import ppg
from colossalai.pipeline.rpc.utils import (color_debug, tensor_shape_list, get_batch_lengths, split_batch, type_detail,
                                           pytree_map, get_real_args_kwargs, use_color_debug)


class Phase(Enum):
    FORWARD = 0
    BACKWARD = 1
    UPDATE = 2


class UniqueKey:
    __slots__ = ('microbatch_id', 'phase')
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
        return f'Key(microbatch_id={self.microbatch_id}, phase={self.phase})'


class WorkItem:
    __slots__ = ('stage_id', 'phase', 'args', 'kwargs', 'output', 'refcount', 'microbatch_id', 'batch_id',
                 'num_microbatches', 'forward_only')

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

    def __init__(self,
                 stage_id,
                 phase,
                 args,
                 kwargs,
                 output,
                 microbatch_id,
                 batch_id,
                 num_microbatches,
                 forward_only,
                 refcount=0) -> None:
        for attr_name in self.__slots__:
            setattr(self, attr_name, locals()[attr_name])


class BackwardCache:
    __slots__ = ('checkpoint', 'stage_input_args', 'stage_input_kwargs', 'stage_outputs')
    checkpoint: bool
    stage_input_args: Tuple[Any]
    stage_input_kwargs: Dict[Any, Any]
    stage_outputs: Tuple[Any]

    def __init__(self,
                 stage_input_args: Tuple[Any],
                 stage_input_kwargs: Dict[Any, Any] = None,
                 stage_outputs: Tuple[Any] = None,
                 checkpoint: bool = False) -> None:
        for arg_name in self.__slots__:
            setattr(self, arg_name, locals()[arg_name])


class WorkerBase(ABC):

    def __init__(self,
                 partition_fn: Callable,
                 partition_args: tuple,
                 pp_rank: int,
                 actual_stage_num: int,
                 num_microbatches: int,
                 device: str,
                 criterion: Callable = None,
                 metric: Callable = None,
                 checkpoint: bool = False,
                 data_process_func: Callable = None) -> None:
        super().__init__()

        self.pp_rank = pp_rank
        self.actual_stage_num = actual_stage_num
        self.num_microbatches = num_microbatches
        self.checkpoint = checkpoint

        if data_process_func is not None:
            self.data_process_func = partial(data_process_func, pp_rank)

        self.device = device
        self._initialize_outstanding_range()

        # variable and const for context managment
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

        # context to maintain loop
        self._initialize_context_container()

        # main loop
        self.main_loop_thread = threading.Thread(target=self._work_loop, name=f'rank_{pp_rank}', daemon=True)
        self.main_loop_thread.start()

    def _get_future_by_device(self):
        return torch.futures.Future(devices=None if self.device in (None, 'cpu') else [self.device])

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

    def _initialize_partition(self):
        partition_fn = self.partition_fn
        partition_args = self.partition_args
        device = self.device
        with self.partition_condition_lock:
            self.module_partition: nn.Module = partition_fn(*partition_args).to(device)
            self.partition_condition_lock.notify_all()

    def sync_global_worker_rrefs(self, pp_rank_to_worker_rref: Dict[int, PyRRef]) -> None:
        assert self.pp_rank_to_worker_rref is None, f"in rank {self.pp_rank}, worker has sync global workers rrefs"
        assert pp_rank_to_worker_rref is not None, "stage_to_workers must be a dict instead of None"
        self.pp_rank_to_worker_rref = pp_rank_to_worker_rref

        # for some schedule need the other worker's info to initialise partition (like Chimera)
        # construction of partition is executed after the registion of pp_rank_to_worker_rref
        self._initialize_partition()

    def get_output_by_key(self, key: UniqueKey) -> Any:
        with self.output_list_condition_lock:
            self.output_list_condition_lock.wait_for(lambda: key in self.output_list)
            output_work_item = self.output_list[key]

        output = output_work_item.output
        if isinstance(output, Future):
            output = output.wait()

        # color_debug(f'rank {self.pp_rank}, output {type(output)}', 'get output', 'red')
        output_work_item.refcount += 1

        # all consumers have been satisfied, the work_item can be released
        with self.output_list_condition_lock:
            if output_work_item.refcount >= len(self.consumer_stage_ids):
                self.output_list.pop(key)
        return output

    def get_parameters(self) -> List[torch.Tensor]:
        return [p for p in self.module_partition.parameters()]

    def get_parameter_gradients(self) -> List[torch.Tensor]:
        return [p.grad for p in self.module_partition.parameters()]

    def get_partition(self):
        with self.partition_condition_lock:
            self.partition_condition_lock.wait_for(lambda: hasattr(self, 'module_partition'))
            return self.module_partition

    def get_partition_state_dict(self):
        with self.partition_condition_lock:
            self.partition_condition_lock.wait_for(lambda: hasattr(self, 'module_partition'))
            return self.module_partition.state_dict()

    def _make_args_kwargs(self, microbatch):
        if isinstance(microbatch, dict):
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
            return args, kwargs
        else:
            raise TypeError(f"Input batch can be only dict, list, tuple or tensor, but receive {type(microbatch)}")

    # just for first pp_rank
    def set_input(self, microbatch_id: int, microbatch: Tuple[Any], forward_only: bool):
        assert self.consumer_stage_ids is not None
        key = UniqueKey(microbatch_id, Phase.FORWARD)
        output = self._get_future_by_device()

        # make args and kwargs
        args, kwargs = self._make_args_kwargs(microbatch)

        work_item = WorkItem(self.pp_rank, Phase.FORWARD, args, kwargs, output, microbatch_id, None,
                             self.num_microbatches, forward_only)
        with self.work_list_condition_lock:
            self.work_list[key] = work_item
            if use_color_debug:
                color_debug(f'rank {self.pp_rank} receive data from dataloader {self._get_store_len()}',
                            'data dispatch', 'magenta')
            self.work_list_condition_lock.notify_all()

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

            work_item = WorkItem(self.pp_rank, Phase.BACKWARD, grad_wrt_loss, {}, output, microbatch_id, None,
                                 self.num_microbatches, False)

            if use_color_debug:
                color_debug(f'rank {self.pp_rank} propose backward', 'data dispatch', 'magenta')

            self.work_list[key] = work_item
            self.work_list_condition_lock.notify_all()

    def subscribe_producer(self, microbatch_id: int, forward_only: bool):
        """
        You should call this function asynchronously
        """
        assert self.producer_stage_ids is not None
        producer_num = len(self.producer_stage_ids)
        assert producer_num > 0, "only stage that has producers can subscribe producers"

        stage_id = self.pp_rank
        subscribe_forward_futures: List[Future] = [None] * producer_num
        output = self._get_future_by_device()

        for i in range(producer_num):
            producer_stage_id = self.producer_stage_ids[i]
            producer_output_key = UniqueKey(microbatch_id, Phase.FORWARD)
            producer_worker_rref = self.pp_rank_to_worker_rref[producer_stage_id]
            subscribe_forward_futures[i] = producer_worker_rref.rpc_async().get_output_by_key(producer_output_key)

        if use_color_debug:
            color_debug(f'rank {self.pp_rank} get {len(subscribe_forward_futures)} futs from its producer',
                        'data dispatch', 'magenta')

        work_item_from_producer = WorkItem(stage_id, Phase.FORWARD, subscribe_forward_futures, {}, output,
                                           microbatch_id, None, self.num_microbatches, forward_only)

        # color_debug(f'rank {self.pp_rank} get value {tensor_shape_list(args)} from fut', 'data dispatch', 'magenta')
        # add work_item to work_list
        with self.work_list_condition_lock:
            key = UniqueKey(microbatch_id, Phase.FORWARD)
            assert key not in self.work_list
            self.work_list[key] = work_item_from_producer
            if use_color_debug:
                color_debug(
                    f'rank_{self.pp_rank} load a new task to its work_list {key} {work_item_from_producer.phase} data: {tensor_shape_list(work_item_from_producer.args)}',
                    'data dispatch', 'magenta')
            self.work_list_condition_lock.notify_all()

    def subscribe_consumer(self, microbatch_id: int):
        """
        You should call this function asynchronously
        """
        assert self.producer_stage_ids is not None
        consumer_num = len(self.consumer_stage_ids)
        assert consumer_num > 0, "only stage that has consumers can subscribe comsumers"

        stage_id = self.pp_rank
        subscribe_backward_futures: List[Future] = [None] * consumer_num
        output = self._get_future_by_device()

        if use_color_debug:
            color_debug(f'rank {self.pp_rank} get {len(subscribe_backward_futures)} futs from its consumer',
                        'data dispatch', 'magenta')

        for i in range(consumer_num):
            consumer_stage_id = self.consumer_stage_ids[i]
            consumer_output_key = UniqueKey(microbatch_id, Phase.BACKWARD)
            consumer_worker_rref = self.pp_rank_to_worker_rref[consumer_stage_id]
            subscribe_backward_futures[i] = consumer_worker_rref.rpc_async().get_output_by_key(consumer_output_key)

        # flatten args
        work_item_from_consumer = WorkItem(stage_id, Phase.BACKWARD, subscribe_backward_futures, {}, output,
                                           microbatch_id, None, self.num_microbatches, False)

        # color_debug(f'rank {self.pp_rank} get value {tensor_shape_list(args)} from fut', 'data dispatch', 'magenta')

        # add work_item to work_list
        with self.work_list_condition_lock:
            key = UniqueKey(microbatch_id, Phase.BACKWARD)
            assert key not in self.work_list
            self.work_list[key] = work_item_from_consumer
            if use_color_debug:
                color_debug(
                    f'rank_{self.pp_rank} load a new task to its work_list {key} {work_item_from_consumer.phase} data: {tensor_shape_list(work_item_from_consumer.args)}',
                    'data dispatch', 'magenta')
            self.work_list_condition_lock.notify_all()

    def _get_producer_consumer(self) -> None:
        rank = self.pp_rank
        assert self.producer_stage_ids is None, f"all the producers of rank {rank} has been subscribed"
        assert self.consumer_stage_ids is None, f"all the consumers of rank {rank} has been subscribed"

        # should be aranged in order, the order of the input of current forward
        self.producer_stage_ids = []
        self.consumer_stage_ids = []

        # Just for demo
        prev_rank = rank - 1
        next_rank = rank + 1
        if prev_rank >= 0:
            self.producer_stage_ids.append(prev_rank)
        if next_rank <= self.actual_stage_num - 1:
            self.consumer_stage_ids.append(next_rank)

    @abstractmethod
    def _get_work_item_key(self) -> UniqueKey:
        """
            this method control the order of the microbatch to consume
        """

    def is_first_stage(self):
        return self.pp_rank == 0

    def is_last_stage(self):
        return self.pp_rank == self.actual_stage_num - 1

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
        data_process_func = getattr(self, 'data_process_func', self._default_data_process_func)
        consume_result = None

        is_first_stage = self.is_first_stage()
        is_last_stage = self.is_last_stage()

        # if self.pp_rank == 0:
        #     print(
        #         f'I am rank_{self.pp_rank} microbatch_id : {microbatch_id} {phase} {self._get_store_len()} | {self.outstanding} {self.outstanding_range}'
        #     )

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
                args = get_real_args_kwargs(args)
                kwargs = get_real_args_kwargs(kwargs)
                args_kwargs = (args, kwargs)
            else:
                args_kwargs = get_real_args_kwargs(args)

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
                # print(f'model{self.pp_rank + 1}(param_sum: {sum([p.sum().item() for p in self.module_partition.parameters()])}) input sum: {args[0].sum().item()} forward output sum: {consume_result.sum().item()}', )

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
                self.microbatch_id_to_backward_cache[microbatch_id] = BackwardCache(stage_input_args,
                                                                                    stage_input_kwargs,
                                                                                    stage_outputs,
                                                                                    checkpoint=use_checkpoint)

            # if not forward_only, do the backward
            if not forward_only:
                if is_last_stage:    # if it is the last stage, trigger backward automatic
                    self._begin_backward(microbatch_id)

        elif phase == Phase.BACKWARD:
            # remind its producer to get data before backward
            if not is_first_stage:
                for stage_id in self.producer_stage_ids:
                    producer_worker_rref = self.pp_rank_to_worker_rref[stage_id]
                    producer_worker_rref.remote().subscribe_consumer(microbatch_id)
            self.backward_times += 1
            self.outstanding -= 1

            assert microbatch_id in self.microbatch_id_to_backward_cache, f"microbatch_id {microbatch_id} not in backward cache"
            backward_cache = self.microbatch_id_to_backward_cache.pop(microbatch_id)

            stage_outputs = backward_cache.stage_outputs
            stage_input_args = backward_cache.stage_input_args
            stage_input_kwargs = backward_cache.stage_input_kwargs
            use_checkpoint = backward_cache.checkpoint

            if use_checkpoint:
                stage_outputs = [self.module_partition(*stage_input_args, **stage_input_kwargs)]

            # take tensor only (for only tensor can do backward)
            stage_outputs_tensors = []
            pytree_map(stage_outputs, stage_outputs_tensors.append, process_types=torch.Tensor)

            # overlap recompute and future.wait
            grad_tensors = get_real_args_kwargs(args)

            # print('rank', self.pp_rank, tensor_shape_list(stage_outputs_tensors), tensor_shape_list(grad_tensors))
            autograd.backward(stage_outputs_tensors, grad_tensors=grad_tensors)

            # collect grad of input tensor
            # there is a hypothesis that node in kwargs cann't be an non-leaf node in graph
            # so we don't need to save the grad of node in kwargs.
            consume_result = []
            if not is_first_stage:
                pytree_map(stage_input_args, lambda x: consume_result.append(x.grad), process_types=torch.Tensor)
                pytree_map(stage_input_kwargs, lambda x: consume_result.append(x.grad), process_types=torch.Tensor)

                # for input_node in stage_input_args:
                #     if isinstance(input_node, torch.Tensor):
                #         consume_result.append(input_node.grad)

        else:
            raise TypeError(f"Unknown phase appears in _consume_work_item_by_phase {phase}")

        return consume_result

    def _get_store_len(self):
        return f'work_list:{len(self.work_list)} output_list:{len(self.output_list)} backward_cache:{len(self.microbatch_id_to_backward_cache)} label_cache:{len(self.microbatch_id_to_labels)}'

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

    def _reset_context(self):
        self.forward_times = 0
        self.backward_times = 0
        self.outstanding = 0
        self._initialize_outstanding_range()

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
                work_item = self.work_list.pop(work_item_key)

            if use_color_debug:
                color_debug(
                    f'rank {self.pp_rank} get a key : {work_item_key} work_item args: {tensor_shape_list(work_item.args)} {self._get_store_len()}',
                    'work loop', 'green')

            with self.output_list_condition_lock:
                # assert work_item_key not in self.output_list
                self.output_list[work_item_key] = work_item
                self.output_list_condition_lock.notify_all()

            consume_result = self._consume_work_item_by_phase(work_item)

            if use_color_debug:
                color_debug(
                    f'rank_{self.pp_rank} [{work_item.phase}] finish consuming, result is {tensor_shape_list(consume_result)} {self._get_store_len()} | {self.work_list.keys()} | {self.output_list.keys()}',
                    'work loop', 'green')

            work_item.output.set_result(consume_result)

            # if is last step in one batch reset context and do step
            if self._is_last_step(work_item):
                self._hook_before_step()
                if hasattr(self, 'optimizer') and not work_item.forward_only:
                    self.step()
                self._reset_context()

    def initialize_optimizer(self, optimizer_class: type, **kwargs):
        self.optimizer: optim.Optimizer = optimizer_class(self.module_partition.parameters(), **kwargs)
        self.step_lock = threading.Lock()
        self.step_lock.acquire()

    def wait_for_step(self):
        self.step_lock.acquire()

    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.step_lock.release()


class PipelineEngineBase(ABC, nn.Module):

    def __init__(self,
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
                 data_process_func: Callable = None) -> None:
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

        self.step_futs: List[Future] = []

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
            assert '<locals>' not in data_process_func.__repr__(), "data_process_func must be a global function"
            assert '<lambda>' not in data_process_func.__repr__(), "data_process_func cannot be a lambda expression"
            sig = inspect.signature(data_process_func)
            assert len(
                sig.parameters
            ) == 2, f"length of data_process_func' arguments must be 2, receive {len(sig.parameters)} arguments instead"

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
            if device[:4] == 'cuda':
                device = f'cuda:{rpc_worker_id}'
            self.pp_rank_to_worker_rref[pp_rank] = rpc.remote(rpc_worker_id,
                                                              worker_type,
                                                              args=(partition_fn, partition_args, pp_rank,
                                                                    actual_stage_num, num_microbatches, device,
                                                                    criterion, metric, checkpoint, data_process_func))

        # let each worker know global worker rref (include itself)
        sync_futs = []
        for pp_rank in self.pp_rank_to_worker_rref:
            fut = self.pp_rank_to_worker_rref[pp_rank].rpc_async().sync_global_worker_rrefs(self.pp_rank_to_worker_rref)
            sync_futs.append(fut)

        for fut in sync_futs:
            fut.wait()

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

    def _consume_constraint(self, microbatch_id: int, forward_only: bool, input_pp_ranks: List[int],
                            output_pp_ranks: List[int], ret_future):
        actual_stage_num = self._get_actual_stage_num()
        use_1F1B = self.use_1F1B
        if microbatch_id >= actual_stage_num:
            if forward_only or not use_1F1B:
                for pp_rank in output_pp_ranks:
                    ret_future[pp_rank][microbatch_id - actual_stage_num].wait()
            else:
                key = UniqueKey(microbatch_id - actual_stage_num, Phase.BACKWARD)
                for pp_rank in input_pp_ranks:
                    worker_rref = self.pp_rank_to_worker_rref[pp_rank]
                    worker_rref.rpc_sync().get_output_by_key(key)

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

    def _subscribe_forward(self, microbatch_id: int, output_pp_ranks: List[int], ret_future: Dict[int, List[Future]]):
        key = UniqueKey(microbatch_id, Phase.FORWARD)
        for pp_rank in output_pp_ranks:
            worker_rref = self.pp_rank_to_worker_rref[pp_rank]
            ret_future[pp_rank][microbatch_id] = worker_rref.rpc_async().get_output_by_key(key)

    def _ensure_backward(self, forward_only: bool, input_pp_ranks: List[int]):
        if not forward_only:
            for pp_rank in input_pp_ranks:
                worker_rref = self.pp_rank_to_worker_rref[pp_rank]
                key = UniqueKey(self.num_microbatches - 1, Phase.BACKWARD)
                worker_rref.rpc_sync().get_output_by_key(key)

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

    def forward_backward(self, batch: torch.Tensor, labels: torch.Tensor = None, forward_only: bool = False):
        batch_lengths = get_batch_lengths(batch)

        if labels is not None and not forward_only:
            assert hasattr(
                self, 'optimizer_class'), "call `initialize_optimizer` to initialize optimizer before forward_backward"

        num_microbatches = self.num_microbatches
        microbatch_size = batch_lengths[0] // num_microbatches
        device = self.device

        # If Chimera mode is used, then rank of down pipeline is excluded from 'input_pp_ranks' or 'output_pp_ranks'
        input_pp_ranks = self.get_input_pp_ranks()
        output_pp_ranks = self.get_output_pp_ranks()

        # a cache to collect data and control flow
        ret_future = self._create_ret_future(output_pp_ranks)

        for microbatch_id in range(num_microbatches):
            # control data input  speed
            # to prevent exceed of wait limitations
            self._consume_constraint(microbatch_id, forward_only, input_pp_ranks, output_pp_ranks, ret_future)
            batch_start = microbatch_size * microbatch_id
            batch_end = batch_start + microbatch_size

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

        if not forward_only and hasattr(self, 'optimizer_class'):
            # wait for all step
            for pp_rank in self.pp_rank_to_worker_rref:
                worker_rref = self.pp_rank_to_worker_rref[pp_rank]
                worker_rref.rpc_sync().wait_for_step()

        return forward_result

    def initialize_optimizer(self, optimizer_class: type, **kwargs):
        self.optimizer_class = optimizer_class
        for pp_rank in self.pp_rank_to_worker_rref:
            worker_rref = self.pp_rank_to_worker_rref[pp_rank]
            worker_rref.remote().initialize_optimizer(optimizer_class, **kwargs)

    def step(self):
        actual_stage_num = self._get_actual_stage_num()
        for pp_rank in range(actual_stage_num):
            worker_rref = self.pp_rank_to_worker_rref[pp_rank]
            fut = worker_rref.rpc_async().step()
            self.step_futs.append(fut)

        for fut in self.step_futs:
            fut.wait()
