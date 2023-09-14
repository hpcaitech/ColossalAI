from colossalai.pipeline.scheduler.worker_state_machine import WorkerStateMachine


class GpipeWorker(WorkerStateMachine):

    def __init__(self, rank, num_minibatches=1):
        super().__init__(rank, num_minibatches)

    def fwd2bwd(self):
        if self.fwd_done():
            return True

    def bwd2fwd(self):
        return False

    def flush(self):
        return True
