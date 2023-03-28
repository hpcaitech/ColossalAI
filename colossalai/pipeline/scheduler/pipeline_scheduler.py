import math

import torch

from colossalai.pipeline.rpc.utils import get_batch_lengths, split_batch
from colossalai.pipeline.scheduler import GpipeWorker


class PipelineScheduler():

    def __init__(self,
                 rank,
                 worker_type,
                 num_stages,
                 num_minibatches,
                 partition_fn,
                 device,
                 checkpoint=False,
                 input_ranks=[0],
                 output_ranks=None):
        self.rank = rank
        self.worker_type = worker_type
        self.num_stages = num_stages
        self.num_minibatches = num_minibatches
        self.partition_fn = partition_fn
        self.device = device
        self.checkpoint = checkpoint
        self.input_ranks = input_ranks
        self.output_ranks = output_ranks if output_ranks else [self.num_stages - 1]

        self.batch = None
        self.labels = None

        self._initialize_worker()

    def is_input_rank(self):
        return self.rank in self.input_ranks

    def is_output_rank(self):
        return self.rank in self.output_ranks

    def set_batch(self, batch: torch.Tensor):
        self.batch = batch

    def set_labels(self, labels: torch.Tensor = None):
        self.labels = labels

    def initialize_optimizer(self, optimizer_class: type, **kwargs):
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = kwargs

    def forward_backward(self, forward_only: bool = False):
        self.worker.set_fwd_only(forward_only)

        if self.is_input_rank() or self.is_output_rank():
            batch_length = get_batch_lengths(self.batch)[0]
            minibatch_size = math.ceil(batch_length / self.num_minibatches)
            device = self.device

            for minibatch_id in range(self.num_minibatches):
                batch_start, batch_end = self._get_batch_offsets(minibatch_size, minibatch_id, batch_length)

                # set input
                if self.is_input_rank():
                    minibatch = split_batch(self.batch, batch_start, batch_end, device)
                    self._set_input(minibatch)
                else:    # set labels
                    if self.labels is not None:
                        minilabels = split_batch(self.labels, batch_start, batch_end, device)
                        self._set_labels(minibatch_id, minilabels)

            if self.is_output_rank():
                for minibatch_id in range(self.num_minibatches):
                    batch_start, batch_end = self._get_batch_offsets(minibatch_size, minibatch_id, batch_length)

        self._wait_for_done(forward_only)

        # collect forward result
        forward_result = self._collect_forward_result()

        return forward_result

    def _initialize_worker(self):
        if self.worker_type == GpipeWorker:
            self.worker = GpipeWorker(self.rank, self.num_minibatches)
        else:
            self.worker = GpipeWorker(self.rank, self.num_minibatches)

        self.worker.set_initial_state("start")
        self.worker.set_device(self.device)
        self.worker.initialize_partition(self.partition_fn, partition_args=(self.rank, self.num_stages))
        self._initialize_communication()
        self.worker.run()

    def _initialize_communication(self):
        self.worker.init_comm(self.rank)

    def _get_batch_offsets(self, minibatch_size, minibatch_id, batch_length):
        batch_start = minibatch_size * minibatch_id
        batch_end = min(batch_start + minibatch_size, batch_length)
        return batch_start, batch_end

    def _set_input(self, minibatch):
        self.worker.add_minibatch(minibatch)

    def _set_labels(self, minibatch_id, minilabels):
        self.worker.add_labels(minibatch_id, minilabels)

    def _wait_for_done(self, forward_only):
        self.worker.wait_for_done(forward_only)

    def _collect_forward_result(self):
        pass
