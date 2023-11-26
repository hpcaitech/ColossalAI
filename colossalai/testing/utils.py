import gc
import random
import re
import socket
from functools import partial
from inspect import signature
from typing import Any, Callable, List

import torch
import torch.multiprocessing as mp
from packaging import version
from colossalai.utils.device import empty_cache, reset_max_memory_allocated, reset_peak_memory_stats, synchronize, reset_max_memory_cached, device_count


def parameterize(argument: str, values: List[Any]) -> Callable:
    """
    This function is to simulate the same behavior as pytest.mark.parameterize. As
    we want to avoid the number of distributed network initialization, we need to have
    this extra decorator on the function launched by torch.multiprocessing.

    If a function is wrapped with this wrapper, non-parametrized arguments must be keyword arguments,
    positional arguments are not allowed.

    Usage::

        # Example 1:
        @parameterize('person', ['xavier', 'davis'])
        def say_something(person, msg):
            print(f'{person}: {msg}')

        say_something(msg='hello')

        # This will generate output:
        # > xavier: hello
        # > davis: hello

        # Example 2:
        @parameterize('person', ['xavier', 'davis'])
        @parameterize('msg', ['hello', 'bye', 'stop'])
        def say_something(person, msg):
            print(f'{person}: {msg}')

        say_something()

        # This will generate output:
        # > xavier: hello
        # > xavier: bye
        # > xavier: stop
        # > davis: hello
        # > davis: bye
        # > davis: stop

    Args:
        argument (str): the name of the argument to parameterize
        values (List[Any]): a list of values to iterate for this argument
    """

    def _wrapper(func):
        def _execute_function_by_param(**kwargs):
            for val in values:
                arg_map = {argument: val}
                partial_func = partial(func, **arg_map)
                partial_func(**kwargs)

        return _execute_function_by_param

    return _wrapper


def rerun_on_exception(exception_type: Exception = Exception, pattern: str = None, max_try: int = 5) -> Callable:
    """
    A decorator on a function to re-run when an exception occurs.

    Usage::

        # rerun for all kinds of exception
        @rerun_on_exception()
        def test_method():
            print('hey')
            raise RuntimeError('Address already in use')

        # rerun for RuntimeError only
        @rerun_on_exception(exception_type=RuntimeError)
        def test_method():
            print('hey')
            raise RuntimeError('Address already in use')

        # rerun for maximum 10 times if Runtime error occurs
        @rerun_on_exception(exception_type=RuntimeError, max_try=10)
        def test_method():
            print('hey')
            raise RuntimeError('Address already in use')

        # rerun for infinite times if Runtime error occurs
        @rerun_on_exception(exception_type=RuntimeError, max_try=None)
        def test_method():
            print('hey')
            raise RuntimeError('Address already in use')

        # rerun only the exception message is matched with pattern
        # for infinite times if Runtime error occurs
        @rerun_on_exception(exception_type=RuntimeError, pattern="^Address.*$")
        def test_method():
            print('hey')
            raise RuntimeError('Address already in use')

    Args:
        exception_type (Exception, Optional): The type of exception to detect for rerun
        pattern (str, Optional): The pattern to match the exception message.
            If the pattern is not None and matches the exception message,
            the exception will be detected for rerun
        max_try (int, Optional): Maximum reruns for this function. The default value is 5.
            If max_try is None, it will rerun forever if exception keeps occurring
    """

    def _match_lines(lines, pattern):
        for line in lines:
            if re.match(pattern, line):
                return True
        return False

    def _wrapper(func):
        def _run_until_success(*args, **kwargs):
            try_count = 0
            assert max_try is None or isinstance(
                max_try, int
            ), f"Expected max_try to be None or int, but got {type(max_try)}"

            while max_try is None or try_count < max_try:
                try:
                    try_count += 1
                    ret = func(*args, **kwargs)
                    return ret
                except exception_type as e:
                    error_lines = str(e).split("\n")
                    if try_count < max_try and (pattern is None or _match_lines(error_lines, pattern)):
                        print("Exception is caught, retrying...")
                        # when pattern is not specified, we always skip the exception
                        # when pattern is specified, we only skip when pattern is matched
                        continue
                    else:
                        print("Maximum number of attempts is reached or pattern is not matched, no more retrying...")
                        raise e

        # Override signature
        # otherwise pytest.mark.parameterize will raise the following error:
        # function does not use argument xxx
        sig = signature(func)
        _run_until_success.__signature__ = sig

        return _run_until_success

    return _wrapper


def rerun_if_address_is_in_use():
    """
    This function reruns a wrapped function if "address already in use" occurs
    in testing spawned with torch.multiprocessing

    Usage::

        @rerun_if_address_is_in_use()
        def test_something():
            ...

    """
    # check version
    torch_version = version.parse(torch.__version__)
    assert torch_version.major >= 1

    # only torch >= 1.8 has ProcessRaisedException
    if torch_version >= version.parse("1.8.0"):
        exception = torch.multiprocessing.ProcessRaisedException
    else:
        exception = Exception

    func_wrapper = rerun_on_exception(exception_type=exception, pattern=".*Address already in use.*")
    return func_wrapper


def skip_if_not_enough_gpus(min_gpus: int):
    """
    This function is used to check the number of available GPUs on the system and
    automatically skip the test cases which require more GPUs.

    Note:
        The wrapped function must have `world_size` in its keyword argument.

    Usage:
        @skip_if_not_enough_gpus(min_gpus=8)
        def test_something():
            # will be skipped if there are fewer than 8 GPUs available
            do_something()

    Arg:
        min_gpus (int): the minimum number of GPUs required to run this test.
    """

    def _wrap_func(f):
        def _execute_by_gpu_num(*args, **kwargs):
            num_avail_gpu = device_count()
            if num_avail_gpu >= min_gpus:
                f(*args, **kwargs)

        return _execute_by_gpu_num

    return _wrap_func


def free_port() -> int:
    """Get a free port on localhost.

    Returns:
        int: A free port on localhost.
    """
    while True:
        port = random.randint(20000, 65000)
        try:
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("localhost", port))
                return port
        except OSError:
            continue


def spawn(func, nprocs=1, **kwargs):
    """
    This function is used to spawn processes for testing.

    Usage:
        # must contains arguments rank, world_size, port
        def do_something(rank, world_size, port):
            ...

        spawn(do_something, nprocs=8)

        # can also pass other arguments
        def do_something(rank, world_size, port, arg1, arg2):
            ...

        spawn(do_something, nprocs=8, arg1=1, arg2=2)

    Args:
        func (Callable): The function to be spawned.
        nprocs (int, optional): The number of processes to spawn. Defaults to 1.
    """
    port = free_port()
    wrapped_func = partial(func, world_size=nprocs, port=port, **kwargs)
    mp.spawn(wrapped_func, nprocs=nprocs)


def clear_cache_before_run():
    """
    This function is a wrapper to clear CUDA and python cache before executing the function.

    Usage:
        @clear_cache_before_run()
        def test_something():
            ...
    """

    def _wrap_func(f):
        def _clear_cache(*args, **kwargs):
            empty_cache()
            reset_peak_memory_stats()
            reset_max_memory_allocated()
            reset_max_memory_cached()
            synchronize()
            gc.collect()
            f(*args, **kwargs)

        return _clear_cache

    return _wrap_func


class DummyDataloader:
    def __init__(self, data_gen_fn: Callable, length: int = 10):
        self.data_gen_fn = data_gen_fn
        self.length = length
        self.step = 0

    def __iter__(self):
        self.step = 0
        return self

    def __next__(self):
        if self.step < self.length:
            self.step += 1
            return self.data_gen_fn()
        else:
            raise StopIteration

    def __len__(self):
        return self.length
