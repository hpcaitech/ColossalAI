import re
from typing import Callable, List, Any
from functools import partial
from inspect import signature


def parameterize(argument: str, values: List[Any]) -> Callable:
    """
    This function is to simulate the same behavior as pytest.mark.parameterize. As
    we want to avoid the number of distributed network initialization, we need to have
    this extra decorator on the function launched by torch.multiprocessing.

    If a function is wrapped with this wrapper, non-paramterized arguments must be keyword arguments,
    positioanl arguments are not allowed.

    Usgae::

        # Example 1:
        @parameterize('person', ['xavier', 'davis'])
        def say_something(person, msg):
            print(f'{person}: {msg}')

        say_something(msg='hello')

        # This will generate output:
        # > xavier: hello
        # > davis: hello

        # Exampel 2:
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
            If max_try is None, it will rerun foreven if exception keeps occurings
    """

    def _match_lines(lines, pattern):
        for line in lines:
            if re.match(pattern, line):
                return True
        return False

    def _wrapper(func):

        def _run_until_success(*args, **kwargs):
            try_count = 0
            assert max_try is None or isinstance(max_try, int), \
                f'Expected max_try to be None or int, but got {type(max_try)}'

            while max_try is None or try_count < max_try:
                try:
                    try_count += 1
                    ret = func(*args, **kwargs)
                    return ret
                except exception_type as e:
                    error_lines = str(e).split('\n')
                    if try_count < max_try and (pattern is None or _match_lines(error_lines, pattern)):
                        print('Exception is caught, retrying...')
                        # when pattern is not specified, we always skip the exception
                        # when pattern is specified, we only skip when pattern is matched
                        continue
                    else:
                        print('Maximum number of attempts is reached or pattern is not matched, no more retrying...')
                        raise e

        # Override signature
        # otherwise pytest.mark.parameterize will raise the following error:
        # function does not use argumetn xxx
        sig = signature(func)
        _run_until_success.__signature__ = sig

        return _run_until_success

    return _wrapper
