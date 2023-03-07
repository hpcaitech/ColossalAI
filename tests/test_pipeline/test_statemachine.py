from colossalai.pipeline.scheduler import GpipeScheduler, WorkerState, WorkerStateMachine

start_state = WorkerState("start", {"start": "fwd"})
fwd_state = WorkerState("fwd", {"fwd2bwd": "bwd", "fwd_done": "end"})
bwd_state = WorkerState("bwd", {"bwd2fwd": "fwd", "flush": "step"})
step_state = WorkerState("step", {"step_done": "end"})
end_state = WorkerState("end", {"next_batch": "start"})


def _step_and_check(state_machine: WorkerStateMachine, target_state: WorkerState):
    state_machine._change_state()
    assert state_machine.current_state == target_state


def _test_gpipe(rank=0, num_minibatches=2):
    # fwd_only
    state_machine: WorkerStateMachine = GpipeScheduler(rank=rank, num_minibatches=num_minibatches, fwd_only=True)
    # start
    assert state_machine.current_state == start_state
    # start->fwd
    _step_and_check(state_machine, fwd_state)
    # fwd->fwd
    state_machine.cur_fwd_id = num_minibatches - 1
    _step_and_check(state_machine, fwd_state)
    # fwd->end
    state_machine.cur_fwd_id = num_minibatches
    _step_and_check(state_machine, end_state)
    # end->start
    _step_and_check(state_machine, start_state)

    # fwd_bwd
    state_machine: WorkerStateMachine = GpipeScheduler(rank=rank, num_minibatches=num_minibatches, fwd_only=False)
    # start
    assert state_machine.current_state == start_state
    # start->fwd
    _step_and_check(state_machine, fwd_state)
    # fwd->fwd
    state_machine.cur_fwd_id = num_minibatches - 1
    _step_and_check(state_machine, fwd_state)
    # fwd->bwd
    state_machine.cur_fwd_id = num_minibatches
    _step_and_check(state_machine, bwd_state)
    # bwd->bwd
    state_machine.cur_bwd_id = num_minibatches - 1
    _step_and_check(state_machine, bwd_state)
    # bwd->step
    state_machine.cur_bwd_id = num_minibatches
    _step_and_check(state_machine, step_state)
    # step->end
    _step_and_check(state_machine, end_state)
    # end->start
    _step_and_check(state_machine, start_state)


def test_statemachine():
    _test_gpipe(rank=0, num_minibatches=2)


if __name__ == "__main__":
    test_statemachine()
