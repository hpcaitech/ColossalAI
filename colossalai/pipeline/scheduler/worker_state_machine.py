import threading
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, Tuple

import torch
from torch import autograd, nn, optim

from colossalai.communication.dag_comm import DAGCommunication
from colossalai.pipeline.scheduler.stage_info import StageInput, StageOutput
from colossalai.pipeline.scheduler.task import Task


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
        self.minibatch_id_to_labels = dict()
        self.minibatch_id_to_backward_cache = dict()

    def init_lock(self):
        self.partition_condition_lock = threading.Condition(threading.Lock())
        self.input_queue_fwd_lock = threading.Condition(threading.Lock())
        self.input_queue_bwd_lock = threading.Condition(threading.Lock())
        self.output_queue_fwd_lock = threading.Condition(threading.Lock())
        self.output_queue_bwd_lock = threading.Condition(threading.Lock())
        self.label_lock = threading.Condition(threading.Lock())

    def init_comm(self, rank):
        self.comm = DAGCommunication(rank)

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
        while True:
            # change state
            next_state = self._change_state()

            # choose action according to state
            res = None
            task = self._get_next_task()
            if task:
                res = task.execute()

            if res:
                self._set_output(res)

    def run(self):
        main_loop_thread = threading.Thread(target=self.loop)
        main_loop_thread.start()
        main_loop_thread.join()

    def add_minibatch(self, minibatch):
        self._push_queue_with_lock(self.input_queue_fwd, self.input_queue_fwd_lock, minibatch)

    def add_labels(self, minibatch_id, minilabels):
        with self.label_lock:
            self.minibatch_id_to_labels[minibatch_id] = minilabels
            self.label_lock.notify_all()

    def wait_for_done(self, forward_only):
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
        task = None
        cur_state = self.get_current_state()
        if cur_state == 'start':
            task = None
        elif cur_state == 'fwd':
            self._wait_for_last_fwds()
            stage_input = self._get_input('fwd', self.cur_fwd_id)
            # TODO use minibatch_id calculated seperately
            task = Task(self._forward, self.cur_fwd_id, *stage_input.args, **stage_input.kwargs)
            self.cur_fwd_id += 1
        elif cur_state == 'bwd':
            self._wait_for_last_bwds()
            stage_input = self._get_input('bwd', self.cur_bwd_id)
            task = Task(self._backward, self.cur_bwd_id, *stage_input.args, **stage_input.kwargs)
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
            stage_output = StageOutput(res)
            self._push_queue_with_lock(self.output_queue_fwd, self.output_queue_fwd_lock, stage_output)
        elif cur_state == 'bwd':
            stage_output = StageOutput(res)
            self._push_queue_with_lock(self.output_queue_bwd, self.output_queue_bwd_lock, stage_output)
        elif cur_state == 'step':
            pass
        elif cur_state == 'end':
            pass
        else:    # wrong state
            pass

    def _push_queue_with_lock(self, queue, lock, event):
        with lock:
            queue.append(event)

    def _pop_queue_with_lock(self, queue, lock):
        with lock:
            return queue.popleft()

    # TODO use aync thread to do the consumer-producer queue.
    def _wait_for_last_fwds(self):
        pass

    # TODO use aync thread to do the consumer-producer queue.
    def _wait_for_last_bwds(self):
        pass
        #stage_input: StageInput = StageInput(self.cur_bwd_id)
        #self._push_queue_with_lock(self.input_queue_bwd, self.input_queue_bwd_lock, stage_input)

    # TODO use aync thread to do the consumer-producer queue.
    def _send_to_next_fwds(self, outputs):
        pass

    # TODO use aync thread to do the consumer-producer queue.
    def _send_to_next_bwds(self, grads):
        pass

    def _gather_grad(self, **args):
        return None

    def _forward(self, minibatch_id, *args, **kwargs):
        # exec fwd
        if self.fwd_only:
            with torch.no_grad():
                stage_outputs = self.module_partition(*args, **kwargs)
        else:
            stage_outputs = self.module_partition(*args, **kwargs)

        # send output
        self._send_to_next_fwds(stage_outputs)

        # save for bwd
        if not self.fwd_only:
            self.minibatch_id_to_backward_cache[minibatch_id] = BackwardCache(args,
                                                                              kwargs,
                                                                              stage_outputs,
                                                                              checkpoint=False)

    def _backward(self, minibatch_id, *args, **kwargs):
        # prepare data
        backward_cache: BackwardCache = self.minibatch_id_to_backward_cache[minibatch_id]
        stage_outputs = backward_cache.stage_outputs
        grad_tensors = kwargs['grad']
        # exec bwd
        autograd.backward(stage_outputs, grad_tensors=grad_tensors)

    def _step(self):
        pass

    def _reset(self):
        pass
