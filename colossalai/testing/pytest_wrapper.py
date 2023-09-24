"""
This file will not be automatically imported by `colossalai.testing`
as this file has a dependency on `pytest`. Therefore, you need to
explicitly import this file `from colossalai.testing.pytest_wrapper import <func>`.from
"""

import os


def run_on_environment_flag(name: str):
    """
    Conditionally run a test based on the environment variable. If this environment variable is set
    to 1, this test will be executed. Otherwise, this test is skipped. The environment variable is default to 0.

    Args:
        name (str): the name of the environment variable flag.

    Usage:
        # in your pytest file
        @run_on_environment_flag(name='SOME_FLAG')
        def test_for_something():
            do_something()

        # in your terminal
        # this will execute your test
        SOME_FLAG=1 pytest test_for_something.py

        # this will skip your test
        pytest test_for_something.py

    """
    try:
        import pytest
    except ImportError:
        raise ImportError(
            "This function requires `pytest` to be installed, please do `pip install pytest` and try again."
        )

    assert isinstance(name, str)
    flag = os.environ.get(name.upper(), "0")

    reason = f"Environment variable {name} is {flag}"
    if flag == "1":
        return pytest.mark.skipif(False, reason=reason)
    else:
        return pytest.mark.skipif(True, reason=reason)
