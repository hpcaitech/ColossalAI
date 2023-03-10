import math

import optim
import torch

from colossalai.pipeline.rpc.utils import get_batch_lengths, split_batch


class PipelineScheduler():

    def __init__(self,
                 rank,
                 worker,
                 num_stages,
                 num_minibatches,
                 partition_fn,
                 device,
                 checkpoint=False,
                 input_ranks=[0]):
        self.rank = rank
        self.worker = worker
        self.num_stages = num_stages
        self.num_minibatches = num_minibatches
        self.partition_fn = partition_fn
        self.device = device
        self.checkpoint = checkpoint
        self.input_ranks = input_ranks

    def is_input_rank(self):
        return self.rank in self.input_ranks

    def initialize_optimizer(self, optimizer_class: type, **kwargs):
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = kwargs

    def forward_backward(self, batch: torch.Tensor, labels: torch.Tensor = None, forward_only: bool = False):
        if not self.is_input_rank():
            return None
        batch_length = get_batch_lengths(batch)[0]

        if labels is not None and not forward_only:
            assert hasattr(
                self, 'optimizer_class'), "call `initialize_optimizer` to initialize optimizer before forward_backward"

        assert batch_length >= self.num_minibatches, "num_microbatches is greater than the size of a batch, which is illegal"
        minibatch_size = math.ceil(batch_length / self.num_minibatches)
        device = self.device

        for minibatch_id in range(self.num_minibatches):
            batch_start, batch_end = self._get_batch_offsets(minibatch_size, minibatch_id, batch_length)

            # set input
            minibatch = split_batch(batch, batch_start, batch_end, device)
            self._set_input(minibatch)

            # set labels
            if labels is not None:
                minilabels = split_batch(labels, batch_start, batch_end, device)
                self._set_labels(minilabels)

            # get data asynchronously
            self._subscribe_forward(minibatch_id)

        self._wait_for_done(forward_only)

        # collect forward result
        forward_result = self._collect_forward_result()

        return forward_result

    def _get_batch_offsets(self, minibatch_size, minibatch_id, batch_length):
        batch_start = minibatch_size * minibatch_id
        batch_end = min(batch_start + minibatch_size, batch_length)
        return batch_start, batch_end

    def _set_input(self, minibatch):
        pass

    def _set_labels(self, minilabels):
        pass

    def _subscribe_forward(self, minibatch_id):
        pass

    def _wait_for_done(self, forward_only):
        pass

    def _collect_forward_result(self):
        pass
