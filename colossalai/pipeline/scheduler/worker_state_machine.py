from abc import ABC, abstractmethod
from collections import deque

from colossalai.pipeline.scheduler.stage_info import StageInput, StageOutput
from colossalai.pipeline.scheduler.task import Task


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
        fwd_only=False,
    ):
        super().__init__()
        self.rank = rank
        self.num_minibatches = num_minibatches
        self.fwd_only = fwd_only

        self.init_state()
        self.init_control_flow()
        self.init_data_queue()

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

    def start_next_batch(self):
        return True

    def run(self):
        while True:
            # change state
            self._change_state()

            # choose action according to state
            res = None
            task = self._get_next_task()
            if task:
                res = task.execute()

            if res:
                self._set_output(res)

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
            if self.start_next_batch():
                next_state = self.get_next_state('next_batch')
            else:
                next_state = None
        else:    # wrong state
            next_state = None

        if next_state:
            self.set_current_state(next_state)

    def _get_next_task(self):
        task = None
        cur_state = self.get_current_state()
        if cur_state == 'start':
            task = None
        elif cur_state == 'fwd':
            stage_input = self._get_input('fwd', self.cur_fwd_id)
            task = Task(self._forward, *stage_input.args, **stage_input.kwargs)
            self.cur_fwd_id += 1
        elif cur_state == 'bwd':
            stage_input = self._get_input('bwd', self.cur_bwd_id)
            task = Task(self._backward, *stage_input.args, **stage_input.kwargs)
            self.cur_bwd_id += 1
        elif cur_state == 'step':
            task = Task(self._step)
        elif cur_state == 'end':
            task = Task(self._reset)
        else:    # wrong state
            task = None
        return task

    def _get_input(self, state, micro_batch_id) -> StageInput:
        # TODO need to add communication framework like p2p
        stage_input = None
        if state == 'fwd':
            stage_input = self.input_queue_fwd.popleft()
        elif state == 'bwd':
            stage_input = self.input_queue_bwd.popleft()
        else:
            stage_input = None
        return stage_input

    def _set_output(self, res):
        cur_state = self.get_current_state()
        if cur_state == 'start':
            pass
        elif cur_state == 'fwd':
            stage_output = StageOutput(res)
            self.output_queue_fwd.append(stage_output)
        elif cur_state == 'bwd':
            stage_output = StageOutput(res)
            self.output_queue_bwd.append(stage_output)
        elif cur_state == 'step':
            pass
        elif cur_state == 'end':
            pass
        else:    # wrong state
            pass

    def _forward(self):
        pass

    def _backward(self):
        pass

    def _step(self):
        pass

    def _reset(self):
        pass
