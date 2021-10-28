import os
from abc import ABC, abstractmethod

import torch
import torch.distributed as dist

from colossalai.communication import all_gather
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.nn.layer._parallel_utilities import _gather
from colossalai.nn.layer.parallel_3d._utils import get_last_group
from colossalai.utils import get_current_device


class Metric(ABC):
    """A basic class of metric collectors. It collects a specific
    metric during training or evaluation and it's always used with 
    :class:`MetricHook` to help it update its states and show the 
    metric. So please use corresponding hook class to make the metric 
    collector works.

    :param epoch_only: Whether the metric only read for the full epoch
    :type epoch_only: bool
    """

    def __init__(self, epoch_only: bool):
        # is the metric only read for the full epoch
        self._epoch_only = epoch_only

    @property
    def epoch_only(self):
        """Returns :attr:`epoch_only`.
        """
        return self._epoch_only

    @abstractmethod
    def reset(self) -> None:
        """Resets the metric to it's initial state.
        By default, this is called at the start of each epoch.
        """
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Updates the metric's state using the passed batch output.
        By default, this is called once for each batch.
        """
        pass

    @abstractmethod
    def get_last_step_value(self):
        """Returns the metric value in the last iteration.
        """
        pass

    @abstractmethod
    def get_accumulated_value(self):
        """Computes the metric based on it's accumulated state.
        By default, this is called at the end of each epoch.

        :return: the actual quantity of interest
        :rtype: Any
        """
        pass

    @staticmethod
    @abstractmethod
    def is_better(a, b) -> bool:
        """Compares a and b, and returns whether a is better than b

        :return: The result of comparison
        :rtype: bool
        """
        pass


class Loss(Metric):
    """A metric collector for loss.

    :param epoch_only: Whether the metric only read for the full epoch
    :type epoch_only: bool
    """

    def __init__(self, epoch_only):
        super().__init__(epoch_only=epoch_only)
        self.last_step_loss = torch.zeros(1, device=get_current_device())
        self.accum_loss = torch.zeros(1, device=get_current_device())
        self.count = 0

    def reset(self) -> None:
        """Sets :attr:`last_step_loss` and :attr:`accum_loss` to zero.
        """
        self.last_step_loss.zero_()
        self.accum_loss.zero_()
        self.count = 0

    def update(self, loss) -> None:
        """Updates :attr:`last_step_loss` and :attr:`accum_loss` with current loss.
        It expects the output has loss.

        :param loss: Current loss of the output
        """
        # expect output to be logits, label and loss
        loss_ = loss.detach()
        self.last_step_loss.copy_(loss_)
        self.accum_loss.add_(loss_)
        self.count += 1

    def get_accumulated_value(self):
        """Returns accumulated loss.
        """
        if gpc.is_initialized(ParallelMode.DATA):
            dist.all_reduce(self.accum_loss, op=dist.ReduceOp.SUM,
                            group=gpc.get_group(ParallelMode.DATA))
            self.accum_loss.div_(gpc.get_world_size(ParallelMode.DATA))

        self.accum_loss.div_(self.count)
        return self.accum_loss.item()

    def get_last_step_value(self):
        """Returns :attr:`last_step_loss`.
        """
        return self.last_step_loss

    def is_better(a, b):
        return a < b


class Accuracy(Metric):
    """A metric collector for accuracy. It only works for classification
    tasks.

    :param epoch_only: Whether the metric only read for the full epoch
    :type epoch_only: bool
    """

    def __init__(self, epoch_only: bool):
        super().__init__(epoch_only=epoch_only)
        self.last_step_sum = torch.zeros(1, device=get_current_device())
        self.last_step_correct = torch.zeros(1, device=get_current_device())
        self.accumulated_sum = torch.zeros(1, device=get_current_device())
        self.accumulated_correct = torch.zeros(1, device=get_current_device())

    def reset(self) -> None:
        self.last_step_sum.zero_()
        self.last_step_correct.zero_()
        self.accumulated_sum.zero_()
        self.accumulated_correct.zero_()

    def update(self, logits, label) -> None:
        """Updates last step accuracy and accumulated accuracy with current logits
        and labels. It expects the output has logits and labels.

        :param logits: The logits output of the model
        :param label: The labels of the input data
        """
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        if isinstance(label, (list, tuple)):
            label = label[0]

        # update
        preds = torch.argmax(logits, dim=-1)
        correct = torch.sum(label == preds)
        self.last_step_sum.fill_(label.size(0))
        self.last_step_correct.fill_(correct)
        self.accumulated_sum += self.last_step_sum
        self.accumulated_correct += self.last_step_correct

    def get_last_step_value(self):
        dist.all_reduce(self.last_step_sum,
                        group=gpc.get_group(ParallelMode.DATA))
        dist.all_reduce(self.last_step_correct,
                        group=gpc.get_group(ParallelMode.DATA))
        return (self.last_step_sum / self.last_step_correct).item()

    def get_accumulated_value(self):
        dist.all_reduce(self.accumulated_sum,
                        group=gpc.get_group(ParallelMode.DATA))
        dist.all_reduce(self.accumulated_correct,
                        group=gpc.get_group(ParallelMode.DATA))
        return (self.accumulated_correct / self.accumulated_sum).item()

    def is_better(a, b) -> bool:
        return a > b


class Accuracy2D(Accuracy):
    """A metric collector for accuracy. It only works for classification
    tasks. This class is the same as :class:`Accuracy` but used in 2D 
    model parallelism.

    :param epoch_only: Whether the metric only read for the full epoch
    :type epoch_only: bool
    """

    def __init__(self, epoch_only: bool):
        super().__init__(epoch_only=epoch_only)

    def update(self, logits, label) -> None:
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        if isinstance(label, (list, tuple)):
            label = label[0]

        logits = _gather(
            logits,
            ParallelMode.PARALLEL_2D_ROW,
            1
        )
        logits = _gather(
            logits,
            ParallelMode.PARALLEL_2D_COL,
            0,
        )
        # update
        preds = torch.argmax(logits, dim=-1)
        correct = torch.sum(label == preds)
        self.last_step_sum.fill_(label.size(0))
        self.last_step_correct.fill_(correct)
        self.accumulated_sum += self.last_step_sum
        self.accumulated_correct += self.last_step_correct


class Accuracy2p5D(Accuracy):
    def __init__(self, epoch_only: bool):
        super().__init__(epoch_only=epoch_only)

    def update(self, logits, label) -> None:
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        if isinstance(label, (list, tuple)):
            label = label[0]

        logits = _gather(
            logits,
            ParallelMode.PARALLEL_2P5D_ROW,
            1
        )
        logits = _gather(
            logits,
            ParallelMode.PARALLEL_2P5D_COL,
            0,
        )
        logits = _gather(
            logits,
            ParallelMode.PARALLEL_2P5D_DEP,
            0,
        )
        # update
        preds = torch.argmax(logits, dim=-1)
        correct = torch.sum(label == preds)
        self.last_step_sum.fill_(label.size(0))
        self.last_step_correct.fill_(correct)
        self.accumulated_sum += self.last_step_sum
        self.accumulated_correct += self.last_step_correct

    def is_better(a, b) -> bool:
        return a > b


class Accuracy3D(Accuracy):
    """A metric collector for accuracy. It only works for classification
    tasks. This class is the same as :class:`Accuracy` but used in 3D 
    model parallelism.

    :param input_parallel_mode: The parallel mode of the input, generally it should be `ParallelMode.PARALLEL_3D_OUTPUT`
    :type input_parallel_mode: `ParallelMode`
    :param weight_parallel_mode: The parallel mode of the weight, generally it should be `ParallelMode.PARALLEL_3D_WEIGHT`
    :type weight_parallel_mode: `ParallelMode`
    :param epoch_only: Whether the metric only read for the full epoch
    :type epoch_only: bool
    """

    def __init__(self, epoch_only, input_parallel_mode, weight_parallel_mode):
        super().__init__(epoch_only=epoch_only)
        self.depth = int(os.environ['DEPTH_3D'])
        self.input_parallel_mode = input_parallel_mode
        self.weight_parallel_mode = weight_parallel_mode
        self.output_parallel_mode = get_last_group(input_parallel_mode,
                                                   weight_parallel_mode)

    def update(self, logits, target):
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        if isinstance(target, (list, tuple)):
            target = target[0]

        batch_size = target.size(0)

        j = gpc.get_local_rank(self.input_parallel_mode)
        i = gpc.get_local_rank(self.weight_parallel_mode)
        target = torch.chunk(target, self.depth, dim=0)[i]
        target = torch.chunk(target, self.depth, dim=0)[j]

        logits = all_gather(logits, -1, self.output_parallel_mode)
        prediction = torch.argmax(logits, dim=-1)
        correct = torch.sum(prediction == target)

        dist.all_reduce(correct, group=gpc.get_group(self.input_parallel_mode))
        dist.all_reduce(correct,
                        group=gpc.get_group(self.weight_parallel_mode))

        self.last_step_sum.fill_(batch_size)
        self.last_step_correct.fill_(correct)
        self.accumulated_sum += self.last_step_sum
        self.accumulated_correct += self.last_step_correct
