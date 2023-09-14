import threading
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, Tuple

import torch
from torch import autograd, nn, optim

from colossalai.communication.dag_comm import DAGCommunication
from colossalai.pipeline.middleware import Partition, Topo
from colossalai.pipeline.scheduler.stage_info import StageInput, StageOutput
from colossalai.pipeline.scheduler.task import Task


class BackwardCache:
    __slots__ = ('checkpoint', 'stage_input_args', 'stage_input_kwargs', 'stage_outputs')
    checkpoint: bool
    stage_input_args: Tuple[Any]
    stage_outputs: Tuple[Any]

    def __init__(self,
                 stage_input_args: Tuple[Any],
                 stage_outputs: Tuple[Any] = None,
                 checkpoint: bool = False) -> None:
        for arg_name in self.__slots__:
            setattr(self, arg_name, locals()[arg_name])


class WorkerState:

    def __init__(self, name, transitions, entry_action=None, exit_action=None):
        self.name = name
        self.transitions = transitions
        self.entry_action = entry_action or (lambda: None)
        self.exit_action = exit_action or (lambda: None)

    def get_next_state(self, event):
        return self.transitions.get(event)

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, WorkerState):
            return __o.name == self.name

        return False


class StateMachine(ABC):

    def __init__(self):
        self.current_state: WorkerState = None
        self.states = {}

    def add_state(self, state: WorkerState):
        self.states[state.name] = state

    def get_state(self, state_name):
        return self.states.get(state_name)

    def set_initial_state(self, state_name):
        self.current_state = self.states[state_name]

    def get_next_state(self, event):
        return self.current_state.get_next_state(event)

    def get_current_state(self):
        return self.current_state.name

    def set_current_state(self, state_name):
        self.current_state = self.get_state(state_name)

    @abstractmethod
    def run(self):
        pass


class WorkerStateMachine(StateMachine):

    def __init__(
        self,
        rank,
        num_minibatches=1,
    ):
        super().__init__()
        self.rank = rank
        self.num_minibatches = num_minibatches
        self.fwd_only = False

        self.init_state()
        self.init_control_flow()
        self.init_data_queue()
        self.init_lock()

        self.set_initial_state("start")

    def init_state(self):
        start_state = WorkerState("start", {"start": "fwd"})
        fwd_state = WorkerState("fwd", {"fwd2bwd": "bwd", "fwd_done": "end"})
        bwd_state = WorkerState("bwd", {"bwd2fwd": "fwd", "flush": "step"})
        step_state = WorkerState("step", {"step_done": "end"})
        end_state = WorkerState("end", {"next_batch": "start"})

        self.add_state(start_state)
        self.add_state(fwd_state)
        self.add_state(bwd_state)
        self.add_state(step_state)
        self.add_state(end_state)

    def init_control_flow(self):
        self.cur_fwd_id = 0
        self.cur_bwd_id = 0

    def init_data_queue(self):
        self.input_queue_fwd = deque()
        self.input_queue_bwd = deque()
        self.output_queue_fwd = deque()
        self.output_queue_bwd = deque()
        self.data_queue = deque()
        self.minibatch_id_to_labels = dict()
        self.minibatch_id_to_backward_cache = dict()

    def init_lock(self):
        self.partition_condition_lock = threading.Condition(threading.Lock())
        self.input_queue_fwd_lock = threading.Condition(threading.Lock())
        self.input_queue_bwd_lock = threading.Condition(threading.Lock())
        self.output_queue_fwd_lock = threading.Condition(threading.Lock())
        self.output_queue_bwd_lock = threading.Condition(threading.Lock())
        self.data_queue_lock = threading.Condition(threading.Lock())
        self.label_lock = threading.Condition(threading.Lock())

    def init_comm(self, rank):
        # initialize comm group with topo
        self.comm = DAGCommunication(rank)

        # add all process groups
        topo: Topo = self._get_topo()
        _, input_partition_ids = self._get_input_partition_ids()
        _, output_partition_ids = self._get_output_partition_ids()

        for input_id in input_partition_ids:
            src_rank = self._partition_id_to_pp_rank(input_id, topo)
            if src_rank is not None:    # src_rank is None means it is the first stage
                self.comm.add_process_group(src_rank, self.rank)

        for output_id in output_partition_ids:
            dst_rank = self._partition_id_to_pp_rank(output_id, topo)
            self.comm.add_process_group(self.rank, dst_rank)

        print(f'rank={self.rank} | pg={self.comm._pg_manager}')

    def set_fwd_only(self, fwd_only=True):
        self.fwd_only = fwd_only

    def set_device(self, device):
        self.device = device

    @abstractmethod
    def fwd2bwd(self):
        pass

    @abstractmethod
    def bwd2fwd(self):
        pass

    @abstractmethod
    def flush(self):
        pass

    def fwd_done(self):
        return self.cur_fwd_id == self.num_minibatches

    def bwd_done(self):
        return self.cur_bwd_id == self.num_minibatches

    def step_done(self):
        return True

    def loop(self):
        cnt = 0
        while True:
            cnt += 1
            # change state
            next_state = self._change_state()

            # choose action according to state
            res = None
            print(f'rank={self.rank} | cnt={cnt}')
            task = self._get_next_task()
            print(f'rank{self.rank} | get task')
            if task:
                res = task.execute()
                print(f'rank{self.rank} | execute task')

            if res:
                self._set_output(res)
                print(f'rank{self.rank} | set res')

    def run(self):
        # main loop
        self.main_loop_thread = threading.Thread(target=self.loop, name=f'rank_{self.rank}')
        self.main_loop_thread.start()

    def add_minibatch(self, minibatch):
        print(f'raw minibatch: {minibatch}')
        args, _ = self._make_args_kwargs(minibatch, merge=True)
        print(f'minibatch: {args}')
        stage_input: StageInput = StageInput(self._get_fwd_minibatch_id(), args)
        self._push_queue_with_lock(self.input_queue_fwd, self.input_queue_fwd_lock, stage_input, notify=True)

    def add_labels(self, minibatch_id, minilabels):
        with self.label_lock:
            self.minibatch_id_to_labels[minibatch_id] = minilabels
            self.label_lock.notify_all()

    def wait_for_done(self, forward_only):
        self.main_loop_thread.join()
        return False

    def initialize_optimizer(self, optimizer_class: type, **kwargs):
        self.optimizer: optim.Optimizer = optimizer_class(self.module_partition.parameters(), **kwargs)

    def initialize_partition(self, partition_fn, partition_args):
        self.partition_fn = partition_fn
        self.partition_args = partition_args
        device = self.device
        with self.partition_condition_lock:
            self.module_partition: nn.Module = partition_fn(*partition_args).to(device)
            self.partition_condition_lock.notify_all()

    def _change_state(self):
        next_state = None
        cur_state = self.get_current_state()
        if cur_state == 'start':
            next_state = self.get_next_state('start')
        elif cur_state == 'fwd':
            # batch end && fwd_only
            if self.fwd_done() and self.fwd_only:
                next_state = self.get_next_state('fwd_done')
            elif self.fwd2bwd():
                next_state = self.get_next_state('fwd2bwd')
            else:
                next_state = None
        elif cur_state == 'bwd':
            if self.bwd_done() and self.flush():
                next_state = self.get_next_state('flush')
            elif self.bwd2fwd():
                next_state = self.get_next_state('bwd2fwd')
            else:
                next_state = None
        elif cur_state == 'step':
            if self.step_done():
                next_state = self.get_next_state('step_done')
            else:
                next_state = None
        elif cur_state == 'end':
            next_state = self.get_next_state('next_batch')
        else:    # wrong state
            next_state = None

        if next_state:
            self.set_current_state(next_state)
        return next_state

    def _get_next_task(self):
        task: Task = None
        cur_state = self.get_current_state()
        if cur_state == 'start':
            task = None
        elif cur_state == 'fwd':
            minibatch_id = self._get_fwd_minibatch_id()
            print(f'rank={self.rank} | before wait fwd')
            self._wait_for_last_fwds(minibatch_id)
            print(f'rank={self.rank} | after wait fwd')
            stage_input = self._get_input('fwd', minibatch_id)
            task = Task(self._forward, minibatch_id, stage_input.args)
            self.cur_fwd_id += 1
        elif cur_state == 'bwd':
            minibatch_id = self._get_bwd_minibatch_id()
            self._wait_for_last_bwds(minibatch_id)
            stage_input = self._get_input('bwd', minibatch_id)
            task = Task(self._backward, minibatch_id, stage_input.args)
            self.cur_bwd_id += 1
        elif cur_state == 'step':
            task = Task(self._step)
        elif cur_state == 'end':
            task = Task(self._reset)
        else:    # wrong state
            task = None
        return task

    def _get_input(self, state, minibatch_id) -> StageInput:
        # TODO need to add communication framework like p2p
        # TODO use minibatch_id to target the data, now it depends on receiving sequence.
        stage_input = None
        if state == 'fwd':
            stage_input = self._pop_queue_with_lock(self.input_queue_fwd, self.input_queue_fwd_lock)
        elif state == 'bwd':
            stage_input = self._pop_queue_with_lock(self.input_queue_bwd, self.input_queue_bwd_lock)
        else:
            stage_input = None
        return stage_input

    def _set_output(self, res):
        cur_state = self.get_current_state()
        if cur_state == 'start':
            pass
        elif cur_state == 'fwd':
            minibatch_id = self._get_fwd_minibatch_id()
            stage_output = StageOutput(minibatch_id, res)
            self._push_queue_with_lock(self.output_queue_fwd, self.output_queue_fwd_lock, stage_output)
        elif cur_state == 'bwd':
            minibatch_id = self._get_bwd_minibatch_id()
            stage_output = StageOutput(minibatch_id, res)
            self._push_queue_with_lock(self.output_queue_bwd, self.output_queue_bwd_lock, stage_output)
        elif cur_state == 'step':
            pass
        elif cur_state == 'end':
            pass
        else:    # wrong state
            pass

    def _get_topo(self):
        if not hasattr(self, '_topo'):
            with self.partition_condition_lock:
                self.partition_condition_lock.wait_for(lambda: hasattr(self, 'module_partition'))
                if hasattr(self.module_partition, '_topo'):
                    self._topo = self.module_partition._topo
                else:
                    self._topo = None
        return self._topo

    def _pp_rank_to_partition_id(self, pp_rank: int, topo: Topo):
        partition_ids = topo.get_mid_partition_ids()
        return partition_ids[pp_rank]

    def _partition_id_to_pp_rank(self, partition_id: int, topo: Topo):
        partition_ids = topo.get_mid_partition_ids()
        for i, id in enumerate(partition_ids):
            if id == partition_id:
                return i

    def _get_input_partition_ids(self, model_input=True):
        topo: Topo = self._get_topo()
        partition_id = self._pp_rank_to_partition_id(self.rank, topo)
        partition: Partition = topo.get_partition_by_id(partition_id)
        model_input_partition_id = topo.get_input_partition_id()
        if model_input:
            input_partition_ids = partition.get_input_partition_ids(input_partition_id=model_input_partition_id)
        else:
            input_partition_ids = partition.get_input_partition_ids()
        return model_input_partition_id, input_partition_ids

    def _get_output_partition_ids(self, model_output=True):
        topo: Topo = self._get_topo()
        partition_id = self._pp_rank_to_partition_id(self.rank, topo)
        partition: Partition = topo.get_partition_by_id(partition_id)
        model_output_partition_id = topo.get_output_partition_id()
        if model_output:
            output_partition_ids = partition.get_output_partition_ids(output_partition_id=model_output_partition_id)
        else:
            output_partition_ids = partition.get_output_partition_ids()
        return model_output_partition_id, output_partition_ids

    def _get_input_offsets_by_index(self, target_index):
        res = []
        topo: Topo = self._get_topo()
        model_input_partition_id, input_partition_ids = self._get_input_partition_ids(model_input=False)
        self_partition_id = self._pp_rank_to_partition_id(self.rank, topo)
        self_partition: Partition = topo.get_partition_by_id(self_partition_id)
        input_vals = self_partition.get_input_vals()

        if self._need_model_input():
            base = 1
        else:
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
                src_stage_id = self._partition_id_to_pp_rank(src_partition_id, topo)
                src_index = base
                for i, id in enumerate(input_partition_ids):
                    stage_id = self._partition_id_to_pp_rank(id, topo)
                    if stage_id == src_stage_id:
                        src_index += i
                        break
            else:    # data from input partition
                src_index = 0
            # when output_len = 1, not iterable
            if target_index == src_index:
                if output_len == 1:
                    res = None    # offset = None to get all outputs
                    return res
                else:
                    res.append(src_offset)
        return res

    def _need_model_input(self):
        need_input = False
        topo: Topo = self._get_topo()
        self_partition_id = self._pp_rank_to_partition_id(self.rank, topo)
        self_partition = topo.get_partition_by_id(self_partition_id)
        partition_inputs = self_partition.get_input_partition_ids()
        model_input_partition_id = topo.get_input_partition_id()
        if model_input_partition_id in partition_inputs:
            need_input = True
        return not self._is_first_stage() and need_input

    def _make_args_kwargs(self, minibatch, merge=False):
        if isinstance(minibatch, dict):
            if merge:
                return list(minibatch.values()), {}
            return [], minibatch
        elif isinstance(minibatch, torch.Tensor):
            return [minibatch], {}
        elif isinstance(minibatch, (tuple, list)):
            args = []
            kwargs = {}
            for arg in minibatch:
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
            raise TypeError(f"Input batch can be only dict, list, tuple or tensor, but receive {type(minibatch)}")

    # TODO reduce communication by cutting res in send, instead of recv
    def _make_args(self, src_partition_id, dst_partition_id, args):
        res = args
        return args

    def _is_first_stage(self):
        return self.rank == 0

    def _push_queue_with_lock(self, queue, lock, event, notify=False):
        with lock:
            queue.append(event)
            if notify:
                lock.notify_all()

    def _pop_queue_with_lock(self, queue, lock):
        with lock:
            return queue.popleft()

    def _get_fwd_minibatch_id(self):
        return self.cur_fwd_id

    def _get_bwd_minibatch_id(self):
        return self.cur_bwd_id

    # TODO use aync thread to do the consumer-producer queue.
    def _wait_for_last_fwds(self, minibatch_id):
        args = []
        # add communication framework like p2p
        topo: Topo = self._get_topo()
        model_input_partition_id, input_partition_ids = self._get_input_partition_ids(model_input=False)

        # input rank need input only
        if self._is_first_stage():
            with self.input_queue_fwd_lock:
                print(f'rank={self.rank} | wait lock')
                self.input_queue_fwd_lock.wait_for(lambda: len(self.input_queue_fwd) > 0)
                input_partition = topo.get_input_partition()
                len_model_input = len(input_partition.get_input_vals())
                if len_model_input == 1:
                    model_input = self.input_queue_fwd.popleft()
                    args = model_input.args
                    if args > 1:
                        model_input.args = [arg for arg in args]
                    self.input_queue_fwd.appendleft(model_input)

            print(f'rank={self.rank} | get lock')
            return
        else:
            res = []
            if self._need_model_input():
                src_rank = self._partition_id_to_pp_rank(model_input_partition_id, topo)
                arg = self.comm.recv(src_rank)
                res.append(arg)
            for input_id in input_partition_ids:
                if model_input_partition_id != input_id:
                    src_rank = self._partition_id_to_pp_rank(input_id, topo)
                    arg = self.comm.recv(src_rank)
                    res.append(arg)

        self_partition_id = self._pp_rank_to_partition_id(self.rank, topo)
        self_partition: Partition = topo.get_partition_by_id(self_partition_id)
        input_vals = self_partition.get_input_vals()

        if self._need_model_input():
            base = 1
        else:
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
                src_stage_id = self._partition_id_to_pp_rank(src_partition_id, topo)
                src_index = base
                for i, id in enumerate(input_partition_ids):
                    stage_id = self._partition_id_to_pp_rank(id, topo)
                    if stage_id == src_stage_id:
                        src_index += i
                        break
            else:    # data from input partition
                src_index = 0

            # when output_len = 1, not iterable
            if output_len == 1:
                target = res[src_index]
            else:
                offsets = self._get_input_offsets_by_index(src_index)
                real_offset = offsets.index(src_offset)
                target = res[src_index][real_offset]
            args.append(target)

        stage_input: StageInput = StageInput(minibatch_id, args)
        self._push_queue_with_lock(self.input_queue_fwd, self.input_queue_fwd_lock, stage_input)

    # TODO use aync thread to do the consumer-producer queue.
    def _wait_for_last_bwds(self, minibatch_id):
        args = ()
        stage_input: StageInput = StageInput(minibatch_id, args)
        self._push_queue_with_lock(self.input_queue_bwd, self.input_queue_bwd_lock, stage_input)

    # TODO use aync thread to do the consumer-producer queue.
    def _send_to_next_fwds(self, outputs):
        topo: Topo = self._get_topo()

        # input stage send input to other stages if necessary
        if self._is_first_stage():
            input_partition = topo.get_input_partition()
            partition_ids = input_partition.get_output_partition_ids()
            for id in partition_ids:
                dst_rank = self._partition_id_to_pp_rank(id, topo)
                args = self._make_args(self._pp_rank_to_partition_id(self.rank), id, outputs)
                self.comm.send(args, dst_rank)
        else:    # send output to next stages
            _, output_partition_ids = self._get_output_partition_ids(model_output=False)
            for output_partition_id in output_partition_ids:
                dst_rank = self._partition_id_to_pp_rank(output_partition_id, topo)
                args = self._make_args(self._pp_rank_to_partition_id(self.rank), output_partition_id, outputs)
                self.comm.send(args, dst_rank)

    # TODO use aync thread to do the consumer-producer queue.
    def _send_to_next_bwds(self, grads):
        pass

    def _gather_grad(self, **args):
        return None

    def _forward(self, minibatch_id, args):
        print(f'rank={self.rank} | forward')
        # exec fwd
        print(args)
        if self.fwd_only:
            with torch.no_grad():
                stage_outputs = self.module_partition(*args)
        else:
            stage_outputs = self.module_partition(*args)

        # send output
        if self._is_first_stage():
            print(f'SEND INPUT TO NEXT STAGE: src_rank={self.rank}')
            self._send_to_next_fwds(args)
            print(f'DONE SEND INPUT TO NEXT STAGE: src_rank={self.rank}')
        print(f'SEND OUTPUT TO NEXT STAGE: src_rank={self.rank}')
        self._send_to_next_fwds(stage_outputs)
        print(f'DONE SEND OUTPUT TO NEXT STAGE: src_rank={self.rank}')

        # save for bwd
        if not self.fwd_only:
            self.minibatch_id_to_backward_cache[minibatch_id] = BackwardCache(args, stage_outputs, checkpoint=False)

    def _backward(
        self,
        minibatch_id,
        args,
    ):
        print(f'rank{self.rank} | backward')
        return 2

        # prepare data
        backward_cache: BackwardCache = self.minibatch_id_to_backward_cache[minibatch_id]
        stage_outputs = backward_cache.stage_outputs
        grad_tensors = kwargs['grad']
        # exec bwd
        autograd.backward(stage_outputs, grad_tensors=grad_tensors)

    def _step(self, minibatch_id, args):
        print(f'rank{self.rank} | step')
        return 2

    def _reset(self, minibatch_id, args):
        print(f'rank{self.rank} | reset')
        exit()
