import threading
from enum import Enum
from typing import List, Any, Tuple, Dict
from abc import ABC

import torch
from torch import nn
import torch.distributed.rpc as rpc
from torch.futures import Future
from torch._C._distributed_rpc import PyRRef
from torch import autograd
from torch import optim
from tqdm import tqdm

from colorama import Back, Style

# config for debug and test
use_color_debug = False
use_progress = False

# TODO:
# 1. replace world_size with other parameters
# 2. adjust to args and kwargs


def color_debug(text, prefix=' ', color='blue'):
    if use_color_debug:
        color = color.upper()
        print(getattr(Back, color), prefix, Style.RESET_ALL, text)


def tensor_shape_list(tensors):
    if isinstance(tensors, torch.Tensor):
        return tensors.shape
    shapes = []
    for t in tensors:
        if hasattr(t, 'shape'):
            shapes.append(t.shape)
        else:
            shapes.append('non tensor')
    return shapes


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
                 'num_microbatches')

    stage_id: int
    phase: Phase
    args: Tuple[Any]
    kwargs: Dict[str, Any]
    output: Future
    microbatch_id: int

    refcount: int

    batch_id: int
    num_microbatches: int

    def __init__(self,
                 stage_id,
                 phase,
                 args,
                 kwargs,
                 output,
                 microbatch_id,
                 batch_id,
                 num_microbatches,
                 refcount=0) -> None:
        for attr_name in self.__slots__:
            setattr(self, attr_name, locals()[attr_name])


class BackwardCache:
    __slots__ = ('checkpoint', 'stage_inputs', 'stage_outputs')
    checkpoint: bool
    stage_inputs: Tuple[Any]
    stage_outputs: Tuple[Any]

    def __init__(self,
                 stage_inputs: List[torch.Tensor],
                 stage_outputs: List[torch.Tensor] = None,
                 checkpoint: bool = False) -> None:
        for arg_name in self.__slots__:
            setattr(self, arg_name, locals()[arg_name])


class RemoteExecutor:

    def __init__(self) -> None:
        pass


class RemoteOptimizer:

    def __init__(self) -> None:
        pass


class Worker:

    def __init__(self,
                 module_partition: nn.Module,
                 pp_rank: int,
                 actual_stage_num: int,
                 num_microbatches: int,
                 max_outstanding: int,
                 device: str,
                 checkpoint: bool = False) -> None:
        super().__init__()
        self.pp_rank = pp_rank
        self.actual_stage_num = actual_stage_num
        self.num_microbatches = num_microbatches
        self.max_outstanding = max_outstanding
        self.outstanding = 0
        self.checkpoint = checkpoint
        self.device = device

        self.future_devices = None if device is None or device == 'cpu' else [device]

        self.pp_rank_to_worker_rref: Dict[int, PyRRef] = None
        self.producer_stage_ids: List[int] = None
        self.consumer_stage_ids: List[int] = None

        # module
        self.module_partition = module_partition.to(device)

        self.debug_list = [None] * num_microbatches

        self.microbatch_id_to_backward_cache: Dict[int, BackwardCache] = dict()

        self.work_list: Dict[UniqueKey, WorkItem] = dict()
        self.output_list: Dict[UniqueKey, WorkItem] = dict()

        # Why must a Lock instead of RLock ?
        # Because RLock cannot be pickled
        self.work_list_condition_lock = threading.Condition(threading.Lock())
        self.output_list_condition_lock = threading.Condition(threading.Lock())

        self.main_loop_thread = threading.Thread(target=self._work_loop, name=f'rank_{pp_rank}', daemon=True)
        self.main_loop_thread.start()

    def _get_future_by_device(self):
        return torch.futures.Future(devices=None if self.device in (None, 'cpu') else [self.device])

    def sync_global_worker_rrefs(self, pp_rank_to_worker_rref: Dict[int, PyRRef]) -> None:
        assert self.pp_rank_to_worker_rref is None, f"in rank {self.pp_rank}, worker has sync global workers rrefs"
        assert pp_rank_to_worker_rref is not None, "stage_to_workers must be a dict instead of None"
        self.pp_rank_to_worker_rref = pp_rank_to_worker_rref

    def get_output_by_key(self, key: UniqueKey) -> Any:
        with self.output_list_condition_lock:
            while key not in self.output_list:
                self.output_list_condition_lock.wait()

            output_work_item = self.output_list[key]

        output = output_work_item.output.wait()
        # color_debug(f'rank {self.pp_rank}, output {type(output)}', 'get output', 'red')
        output_work_item.refcount += 1

        # all consumers have been satisfied, the work_item can be released
        with self.output_list_condition_lock:
            if output_work_item.refcount == len(self.consumer_stage_ids):
                self.output_list.pop(key)

        return output

    def get_parameters(self) -> List[torch.Tensor]:
        return [p for p in self.module_partition.parameters()]

    def get_parameter_gradients(self) -> List[torch.Tensor]:
        return [p.grad for p in self.module_partition.parameters()]

    # just for first pp_rank
    def set_input(self, microbatch_id: int, microbatch: Tuple[Any]):
        with self.work_list_condition_lock:
            assert self.consumer_stage_ids is not None
            consumer_num = len(self.consumer_stage_ids)
            key = UniqueKey(microbatch_id, Phase.FORWARD)
            output = self._get_future_by_device()
            args = [microbatch] if isinstance(microbatch, torch.Tensor) else microbatch

            work_item = WorkItem(self.pp_rank, Phase.FORWARD, args, {}, output, microbatch_id, None,
                                 self.num_microbatches, consumer_num)
            self.work_list[key] = work_item

            color_debug(f'rank {self.pp_rank} receive data from dataloader', 'data dispatch', 'magenta')

            self.work_list_condition_lock.notify_all()

    # just for last pp_rank
    def _begin_backward(self, microbatch_id: int):
        with self.work_list_condition_lock:
            assert self.producer_stage_ids is not None
            producer_num = len(self.producer_stage_ids)
            key = UniqueKey(microbatch_id, Phase.BACKWARD)
            output = self._get_future_by_device()
            grad_wrt_loss = torch.tensor(1, device=self.device)

            work_item = WorkItem(self.pp_rank, Phase.BACKWARD, grad_wrt_loss, {}, output, microbatch_id, None,
                                 self.num_microbatches, producer_num)

            color_debug(f'rank {self.pp_rank} propose backward', 'data dispatch', 'magenta')

            self.work_list[key] = work_item
            self.work_list_condition_lock.notify_all()

    def subscribe_producer(self, microbatch_id: int):
        """
        You should call this function asynchronously
        """
        assert self.producer_stage_ids is not None
        producer_num = len(self.producer_stage_ids)
        consumer_num = len(self.consumer_stage_ids)
        assert producer_num > 0, "only stage that has producers can subscribe producers"

        stage_id = self.pp_rank

        subscribe_forward_futures: List[Future] = [None] * producer_num
        output = self._get_future_by_device()

        for i in range(producer_num):
            producer_stage_id = self.producer_stage_ids[i]
            producer_output_key = UniqueKey(microbatch_id, Phase.FORWARD)
            producer_worker_rref = self.pp_rank_to_worker_rref[producer_stage_id]
            subscribe_forward_futures[i] = producer_worker_rref.rpc_async().get_output_by_key(producer_output_key)

        color_debug(f'rank {self.pp_rank} get {len(subscribe_forward_futures)} futs from its producer', 'data dispatch',
                    'magenta')

        args = []
        for i in range(producer_num):
            producer_args = subscribe_forward_futures[i].wait()
            args.extend(producer_args)

        # TODO : not only args
        work_item_from_producer = WorkItem(stage_id, Phase.FORWARD, args, {}, output, microbatch_id, None,
                                           self.num_microbatches, consumer_num)

        color_debug(f'rank {self.pp_rank} get value {tensor_shape_list(args)} from fut', 'data dispatch', 'magenta')
        # add work_item to work_list
        with self.work_list_condition_lock:
            key = UniqueKey(microbatch_id, Phase.FORWARD)
            assert key not in self.work_list
            self.work_list[key] = work_item_from_producer
            color_debug(
                f'rank_{self.pp_rank} load a new task to its work_list {key} {work_item_from_producer.phase} data: {tensor_shape_list(work_item_from_producer.args)}',
                'data dispatch', 'magenta')
            self.work_list_condition_lock.notify_all()

    def subscribe_consumer(self, microbatch_id: int):
        """
        You should call this function asynchronously
        """
        assert self.producer_stage_ids is not None
        producer_num = len(self.producer_stage_ids)
        consumer_num = len(self.consumer_stage_ids)
        assert consumer_num > 0, "only stage that has consumers can subscribe comsumers"

        # TODO : is this right?
        stage_id = self.pp_rank

        subscribe_backward_futures: List[Future] = [None] * consumer_num
        output = self._get_future_by_device()

        color_debug(f'rank {self.pp_rank} get {len(subscribe_backward_futures)} futs from its consumer',
                    'data dispatch', 'magenta')

        for i in range(consumer_num):
            consumer_stage_id = self.consumer_stage_ids[i]
            consumer_output_key = UniqueKey(microbatch_id, Phase.BACKWARD)
            consumer_worker_rref = self.pp_rank_to_worker_rref[consumer_stage_id]
            subscribe_backward_futures[i] = consumer_worker_rref.rpc_async().get_output_by_key(consumer_output_key)

        args = []
        for i in range(consumer_num):
            consumer_args = subscribe_backward_futures[i].wait()
            args.extend(consumer_args)

        # flatten args
        work_item_from_consumer = WorkItem(stage_id, Phase.BACKWARD, args, {}, output, microbatch_id, None,
                                           self.num_microbatches, producer_num)

        color_debug(f'rank {self.pp_rank} get value {tensor_shape_list(args)} from fut', 'data dispatch', 'magenta')

        # add work_item to work_list
        with self.work_list_condition_lock:
            key = UniqueKey(microbatch_id, Phase.BACKWARD)
            assert key not in self.work_list
            self.work_list[key] = work_item_from_consumer
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

    def _get_work_item_key(self) -> UniqueKey:
        with self.work_list_condition_lock:
            while len(self.work_list) == 0:
                self.work_list_condition_lock.wait()

            # execute backward first (if backward phase in work_list)
            select_work_list_key = None
            for key in self.work_list:
                work_item = self.work_list[key]
                if work_item.phase == Phase.FORWARD and \
                    self.max_outstanding is not None and \
                    self.outstanding >= self.max_outstanding:
                    continue
                else:
                    if select_work_list_key is not None and \
                        select_work_list_key.phase == Phase.FORWARD and \
                        key.phase == Phase.BACKWARD:
                        continue

                    if select_work_list_key is None:
                        select_work_list_key = key
                    else:
                        phase_pair = (select_work_list_key.phase, key.phase)
                        # choose forward first
                        if phase_pair == (Phase.BACKWARD, Phase.FORWARD):
                            select_work_list_key = key
                        elif phase_pair == (Phase.FORWARD, Phase.BACKWARD):
                            continue
                        # choose work_item which has a smaller microbactch_id first
                        elif key.microbatch_id < select_work_list_key.microbatch_id:
                            select_work_list_key = key

        return select_work_list_key

    def _consume_work_item_by_phase(self, work_item: WorkItem):
        phase = work_item.phase
        args = work_item.args
        kwargs = work_item.kwargs
        microbatch_id = work_item.microbatch_id
        consume_result = None

        # if self.pp_rank == 0:
        #     print(f"I am rank_{self.pp_rank} microbatch_id : {microbatch_id}", work_item.phase, len(self.work_list))

        # color_debug(f'rank_{self.pp_rank} enter consume', 'consume', 'blue')

        if phase == Phase.FORWARD:
            self.outstanding += 1

            # TODO : more elegant ?
            for i in range(len(args)):
                arg_obj = args[i]
                if isinstance(arg_obj, torch.Tensor) and not arg_obj.requires_grad:
                    args[i] = arg_obj.requires_grad_()

            # TODO : use process manager to acquire rank info later
            is_last_stage = (self.pp_rank == self.actual_stage_num - 1)

            # last stage doesn't need to do checkpoint, for it will do backward instantly
            if self.checkpoint and not is_last_stage:
                with torch.no_grad():
                    consume_result = self.module_partition(*args, **kwargs)
                stage_outputs = None
                stage_inputs = args
                self.microbatch_id_to_backward_cache[microbatch_id] = BackwardCache(stage_inputs,
                                                                                    stage_outputs,
                                                                                    checkpoint=True)
            else:
                consume_result = self.module_partition(*args, **kwargs)

                stage_outputs = consume_result
                stage_inputs = args
                self.microbatch_id_to_backward_cache[microbatch_id] = BackwardCache(stage_inputs,
                                                                                    stage_outputs,
                                                                                    checkpoint=False)

            consume_result = [consume_result] if isinstance(consume_result, torch.Tensor) else consume_result

            # if it is the last stage, trigger backward automatic
            if is_last_stage:
                self._begin_backward(microbatch_id)

        elif phase == Phase.BACKWARD:
            self.outstanding -= 1
            assert microbatch_id in self.microbatch_id_to_backward_cache, f"microbatch_id {microbatch_id} not in backward cache"
            backward_cache = self.microbatch_id_to_backward_cache.pop(microbatch_id)

            stage_outputs = backward_cache.stage_outputs
            stage_inputs = backward_cache.stage_inputs
            grad_tensors = args

            use_checkpoint = backward_cache.checkpoint

            if use_checkpoint:
                stage_outputs = [self.module_partition(*stage_inputs)]

            autograd.backward(stage_outputs, grad_tensors=grad_tensors)

            # collect grad of input tensor
            consume_result = []
            for input_node in stage_inputs:
                if isinstance(input_node, torch.Tensor):
                    consume_result.append(input_node.grad)

        else:
            raise TypeError(f"Unknown phase appears in _consume_work_item_by_phase {phase}")

        return consume_result

    # do the main loop to consume ready_list
    def _work_loop(self):
        # for init
        self._get_producer_consumer()

        # main loop
        while True:
            work_item_key = self._get_work_item_key()
            if work_item_key is None:
                continue

            # move current work item to output_list to activate subscribe in advance
            with self.work_list_condition_lock:
                work_item = self.work_list.pop(work_item_key)

            color_debug(
                f'rank {self.pp_rank} get a key : {work_item_key} work_item args: {tensor_shape_list(work_item.args)}',
                'work loop', 'green')

            with self.output_list_condition_lock:
                # assert work_item_key not in self.output_list
                self.output_list[work_item_key] = work_item
                self.output_list_condition_lock.notify_all()

            consume_result = self._consume_work_item_by_phase(work_item)

            color_debug(
                f'rank_{self.pp_rank} [{work_item.phase}] finish consuming, result is {tensor_shape_list(consume_result)}',
                'work loop', 'green')
            work_item.output.set_result(consume_result)

    def initialize_optimizer(self, optimizer_class: type, **kwargs):
        self.optimizer: optim.Optimizer = optimizer_class(self.module_partition.parameters(), **kwargs)

    def step(self):
        assert hasattr(self, "optimizer"), "call initialize_optimizer first before you call step!"
        self.work_list.clear()
        self.output_list.clear()
        self.microbatch_id_to_backward_cache.clear()

        self.optimizer.step()
        self.optimizer.zero_grad()


# TODO
# 1. chunk
# 2. checkpoint
class PipelineEngineBase(ABC, nn.Module):

    def __init__(self,
                 module_partitions,
                 stage_num,
                 num_microbatches,
                 device: str,
                 max_outstanding=None,
                 chunk: int = 1,
                 use_interleave: bool = False,
                 checkpoint: bool = False) -> None:
        super().__init__()
        self.module_partitions: List[nn.Module] = module_partitions
        self.chunk = chunk
        self.num_microbatches = num_microbatches
        self.device = device
        self.max_outstanding = max_outstanding
        self.stage_num = stage_num
        self.checkpoint = checkpoint
        self.use_interleave = use_interleave

        self.pp_rank_to_worker_rref: Dict[int, PyRRef] = dict()

        self._check_argument()
        self._create_pp_rank_to_rpc_worker_id()
        self._init_worker()

    def _check_argument(self):
        self.virtual_stage_num = self.stage_num * self.chunk

        assert self.stage_num <= torch.cuda.device_count(), "stage_num must be smaller than device count!"
        assert self.virtual_stage_num == len(
            self.module_partitions), "stage_num * chunk must be equal to length of model partition!"
        if self.use_interleave:
            assert self.num_microbatches % self.stage_num == 0, "if you use interleaving strategy, make sure 'num_microbatches' is a multiple of stage_num!"

    def _get_actual_stage_num(self):
        return self.stage_num if self.chunk == 1 else self.virtual_stage_num

    def _create_pp_rank_to_rpc_worker_id(self):
        """create a map from model partition to stage_id, which is useful when use_interleave is True.
        e.g. If a model is splited into 4 parts, which means len(self.module_partitions) == 3. 
        stage_num is 2, chunk is 2, then pp_rank_to_rpc_worker_id = [0, 1, 0, 1], that means first and third part
        of partitions will be moved to device 0 and the others to device 1

        """
        stage_num = self.stage_num
        actual_stage_num = self._get_actual_stage_num()
        self.pp_rank_to_rpc_worker_id = [0] * actual_stage_num
        for pp_rank in range(actual_stage_num):
            self.pp_rank_to_rpc_worker_id[pp_rank] = pp_rank % stage_num

    def _init_worker(self):
        actual_stage_num = self._get_actual_stage_num()

        max_outstanding = self.max_outstanding
        checkpoint = self.checkpoint
        num_microbatches = self.num_microbatches
        device = self.device

        for pp_rank in range(actual_stage_num):
            module_partition = self.module_partitions[pp_rank]
            rpc_worker_id = self.pp_rank_to_rpc_worker_id[pp_rank]
            if device[:4] == 'cuda':
                device = f'cuda:{rpc_worker_id}'
            self.pp_rank_to_worker_rref[pp_rank] = rpc.remote(rpc_worker_id,
                                                              Worker,
                                                              args=(module_partition, pp_rank, actual_stage_num,
                                                                    num_microbatches, max_outstanding, device,
                                                                    checkpoint))

        # let each worker know global worker rref (include itself)
        for pp_rank in range(actual_stage_num):
            self.pp_rank_to_worker_rref[pp_rank].rpc_sync().sync_global_worker_rrefs(self.pp_rank_to_worker_rref)

    def remote_parameters(self) -> Dict[int, List[torch.Tensor]]:
        parameters = {}
        for stage_id in self.pp_rank_to_worker_rref:
            parameters[stage_id] = []
            worker_rref = self.pp_rank_to_worker_rref[stage_id]
            for p in worker_rref.rpc_sync().get_parameters():
                parameters[stage_id].append(p)
        return parameters

    def remote_grad(self) -> Dict[int, List[torch.Tensor]]:
        grads = {}
        for stage_id in self.pp_rank_to_worker_rref:
            grads[stage_id] = []
            worker_rref = self.pp_rank_to_worker_rref[stage_id]
            for grad in worker_rref.rpc_sync().get_parameter_gradients():
                grads[stage_id].append(grad)
        return grads

    def forward_backward(self, batch: torch.Tensor):
        first_stage_worker = self.pp_rank_to_worker_rref[0]
        microbatch_size = len(batch) // self.num_microbatches
        actual_stage_num = self._get_actual_stage_num()

        microbatch_iter = range(self.num_microbatches)
        if use_progress:
            microbatch_iter = tqdm(microbatch_iter)

        for microbatch_id in microbatch_iter:
            microbatch = batch[microbatch_size * microbatch_id:microbatch_size * (microbatch_id + 1)]

            # forward subscribe asynchronously
            for pp_rank in range(1, actual_stage_num, 1):
                worker_rref = self.pp_rank_to_worker_rref[pp_rank]
                worker_rref.rpc_async().subscribe_producer(microbatch_id)

            # backward subscribe asynchronously
            for pp_rank in range(actual_stage_num - 2, -1, -1):
                worker_rref = self.pp_rank_to_worker_rref[pp_rank]
                worker_rref.rpc_async().subscribe_consumer(microbatch_id)

            # run one microbatch
            first_stage_worker.rpc_sync().set_input(microbatch_id, microbatch)

        # wait forward
        # TODO : all the node to output
        forward_result = None
        last_worker_rref = self.pp_rank_to_worker_rref[actual_stage_num - 1]
        for microbatch_id in range(self.num_microbatches):
            key = UniqueKey(microbatch_id, Phase.FORWARD)
            ret = last_worker_rref.rpc_sync().get_output_by_key(key)
            if forward_result is None:
                forward_result = [[]] * len(ret)
            for i in range(len(forward_result)):
                forward_result[i].append(ret[i])

        # wait for last backward in rank0
        key = UniqueKey(self.num_microbatches - 1, Phase.BACKWARD)
        first_stage_worker.rpc_sync().get_output_by_key(key)
        return forward_result

    def initialize_optimizer(self, optimizer_class: type, **kwargs):
        actual_stage_num = self._get_actual_stage_num()
        for pp_rank in range(actual_stage_num):
            worker_rref = self.pp_rank_to_worker_rref[pp_rank]
            worker_rref.rpc_sync().initialize_optimizer(optimizer_class, **kwargs)

    def step(self):
        step_futs: List[Future] = []
        actual_stage_num = self._get_actual_stage_num()
        for pp_rank in range(actual_stage_num):
            worker_rref = self.pp_rank_to_worker_rref[pp_rank]
            fut = worker_rref.rpc_async().step()
            step_futs.append(fut)

        # wait for all optimizers
        for fut in step_futs:
            fut.wait()


class FillDrainPipelineEngine(PipelineEngineBase):

    def __init__(self,
                 module_partitions: List[nn.Module],
                 stage_num: int,
                 num_microbatches: int,
                 device: str,
                 chunk: int = 1,
                 use_interleave: bool = False,
                 checkpoint: bool = False) -> None:
        max_outstanding = None
        super().__init__(module_partitions, stage_num, num_microbatches, device, max_outstanding, chunk, use_interleave,
                         checkpoint)


class OneFOneBPipelineEngine(PipelineEngineBase):

    def __init__(self,
                 module_partitions: List[nn.Module],
                 stage_num: int,
                 num_microbatches: int,
                 device: str,
                 max_outstanding=None,
                 chunk: int = 1,
                 use_interleave: bool = False,
                 checkpoint: bool = False) -> None:
        if max_outstanding is None:
            max_outstanding = len(module_partitions)
        super().__init__(module_partitions, stage_num, num_microbatches, device, max_outstanding, chunk, use_interleave,
                         checkpoint)
