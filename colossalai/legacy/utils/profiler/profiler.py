import gzip
import json
import os
import tempfile
from typing import Any, Callable, Iterable, List, Optional

from torch.autograd import ProfilerActivity
from torch.profiler import profile as torch_profile
from torch.profiler.profiler import ProfilerAction

from colossalai.legacy.engine import Engine
from colossalai.legacy.utils.profiler.extention import ProfilerExtension
from colossalai.legacy.utils.profiler.stateful_tensor_mem_extention import StatefulTensorMemoryProfilerExtention
from colossalai.logging import get_dist_logger


class profile(torch_profile):
    """Profiler context manager.

    Args:
        activities (iterable): list of activity groups (CPU, CUDA) to use in profiling, supported values:
            ``torch.profiler.ProfilerActivity.CPU``, ``torch.profiler.ProfilerActivity.CUDA``.
            Default value: ProfilerActivity.CPU and (when available) ProfilerActivity.CUDA.
        schedule (callable): callable that takes step (int) as a single parameter and returns
            ``ProfilerAction`` value that specifies the profiler action to perform at each step.
        on_trace_ready (callable): callable that is called at each step when ``schedule``
            returns ``ProfilerAction.RECORD_AND_SAVE`` during the profiling.
        engine (Optional[Engine], optional): An ``Engine`` instance. Defaults to None.
        record_shapes (bool): save information about operator's input shapes.
        profile_memory (bool): track tensor memory allocation/deallocation.
        with_stack (bool): record source information (file and line number) for the ops.
        with_flops (bool): use formula to estimate the FLOPs (floating point operations) of specific operators
            (matrix multiplication and 2D convolution).
        with_modules (bool): record module hierarchy (including function names)
            corresponding to the callstack of the op. e.g. If module A's forward call's
            module B's forward which contains an aten::add op,
            then aten::add's module hierarchy is A.B
            Note that this support exist, at the moment, only for TorchScript models
            and not eager mode models.
        profile_stateful_tensor_memory (bool): track stateful tensor memory usage. ``engine`` must not be None if you enable this.

    .. note::
        Use :func:`~torch.profiler.schedule` to generate the callable schedule.
        Non-default schedules are useful when profiling long training jobs
        and allow the user to obtain multiple traces at the different iterations
        of the training process.
        The default schedule simply records all the events continuously for the
        duration of the context manager.

    .. note::
        Use :func:`~torch.profiler.tensorboard_trace_handler` to generate result files for TensorBoard:

        ``on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name)``

        After profiling, result files can be found in the specified directory. Use the command:

        ``tensorboard --logdir dir_name``

        to see the results in TensorBoard.
        For more information, see
        `PyTorch Profiler TensorBoard Plugin <https://github.com/pytorch/kineto/tree/master/tb_plugin>`__

    .. note::
        Enabling shape and stack tracing results in additional overhead.
        When record_shapes=True is specified, profiler will temporarily hold references to the tensors;
        that may further prevent certain optimizations that depend on the reference count and introduce
        extra tensor copies.

    Examples:

    .. code-block:: python

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as p:
            code_to_profile()
        print(p.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=-1))

    Using the profiler's ``schedule``, ``on_trace_ready`` and ``step`` functions:

    .. code-block:: python

        # Non-default profiler schedule allows user to turn profiler on and off
        # on different iterations of the training loop;
        # trace_handler is called every time a new trace becomes available
        def trace_handler(prof):
            print(prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=-1))
            # prof.export_chrome_trace("/tmp/test_trace_" + str(prof.step_num) + ".json")

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],

            # In this example with wait=1, warmup=1, active=2,
            # profiler will skip the first step/iteration,
            # start warming up on the second, record
            # the third and the forth iterations,
            # after which the trace will become available
            # and on_trace_ready (when set) is called;
            # the cycle repeats starting with the next step

            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=2),
            on_trace_ready=trace_handler
            # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
            # used when outputting for tensorboard
            ) as p:
                for iter in range(N):
                    code_iteration_to_profile(iter)
                    # send a signal to the profiler that the next iteration has started
                    p.step()
    """

    def __init__(
        self,
        *,
        activities: Optional[Iterable[ProfilerActivity]] = None,
        schedule: Optional[Callable[[int], ProfilerAction]] = None,
        on_trace_ready: Optional[Callable[..., Any]] = None,
        engine: Optional[Engine] = None,
        record_shapes: bool = False,
        profile_memory: bool = False,
        with_stack: bool = False,
        with_flops: bool = False,
        with_modules: bool = False,
        profile_stateful_tensor_memory: bool = False,
    ) -> None:
        super().__init__(
            activities=activities,
            schedule=schedule,
            on_trace_ready=on_trace_ready,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
            with_flops=with_flops,
            with_modules=with_modules,
        )
        self._logger = get_dist_logger()
        self.extentions: List[ProfilerExtension] = []
        if profile_stateful_tensor_memory:
            if engine is None:
                self._logger.warning('Ignore "profile_model_data" since engine is None', ranks=[0])
            else:
                self.extentions.append(StatefulTensorMemoryProfilerExtention(engine))

    def prepare_trace(self) -> None:
        if hasattr(super(), "prepare_trace"):
            super().prepare_trace()
        elif hasattr(super(), "_start_warmup"):
            super()._start_warmup()
        for ext in self.extentions:
            ext.prepare_trace()

    def _start_warmup(self):
        self.prepare_trace()

    def start_trace(self):
        if hasattr(super(), "_start_trace"):
            super()._start_trace()
        elif hasattr(super(), "start_trace"):
            super().start_trace()
        for ext in self.extentions:
            ext.start_trace()

    def _start_trace(self):
        self.start_trace()

    def stop_trace(self):
        if hasattr(super(), "_stop_trace"):
            super()._stop_trace()
        elif hasattr(super(), "stop_trace"):
            super().stop_trace()
        for ext in self.extentions:
            ext.stop_trace()

    def _stop_trace(self):
        self.stop_trace()

    def export_chrome_trace(self, path: str):
        """
        Exports the collected trace in Chrome JSON format.
        """
        assert self.profiler
        fp = tempfile.NamedTemporaryFile("w+t", suffix=".json", delete=False)
        fp.close()
        retvalue = self.profiler.export_chrome_trace(fp.name)
        with open(fp.name) as fin:
            trace = json.load(fin)
            for ext in self.extentions:
                trace = ext.extend_chrome_trace(trace)
            open_func = gzip.open if path.endswith(".gz") else open
            with open_func(path, "wt") as fout:
                json.dump(trace, fout)

        os.remove(fp.name)
        return retvalue
