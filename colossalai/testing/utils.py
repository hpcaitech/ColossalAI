from typing import List, Any
from functools import partial


def parameterize(argument: str, values: List[Any]):
    """
    This function is to simulate the same behavior as pytest.mark.parameterize. As
    we want to avoid the number of distributed network initialization, we need to have
    this extra decorator on the function launched by torch.multiprocessing.

    If a function is wrapped with this wrapper, non-paramterized arguments must be keyword arguments,
    positioanl arguments are not allowed.

    Example 1:

    @parameterize('person', ['xavier', 'davis'])
    def say_something(person, msg):
        print(f'{person}: {msg}')

    say_something(msg='hello')

    This will generate output:
    > xavier: hello
    > davis: hello


    Exampel 2:

    @parameterize('person', ['xavier', 'davis'])
    @parameterize('msg', ['hello', 'bye', 'stop'])
    def say_something(person, msg):
        print(f'{person}: {msg}')

    say_something()

    This will generate output:
    > xavier: hello
    > xavier: bye
    > xavier: stop
    > davis: hello
    > davis: bye
    > davis: stop
    """

    def _wrapper(func):
        def _execute_function_by_param(**kwargs):
            for val in values:
                arg_map = {argument: val}
                partial_func = partial(func, **arg_map)
                partial_func(**kwargs)
        return _execute_function_by_param
    return _wrapper
