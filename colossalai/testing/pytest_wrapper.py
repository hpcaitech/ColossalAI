import pytest
import os


def run_on_environment_flag(name: str):
    """
    Conditionally run a test based on the environment variable. If this environment variable is set
    to 1, this test will be executed. Otherwise, this test is skipped. The environment variable is default to 0.
    """
    assert isinstance(name, str)
    flag = os.environ.get(name.upper(), '0')

    reason = f'Environment varialbe {name} is {flag}'
    if flag == '1':
        return pytest.mark.skipif(False, reason=reason)
    else:
        return pytest.mark.skipif(True, reason=reason)
