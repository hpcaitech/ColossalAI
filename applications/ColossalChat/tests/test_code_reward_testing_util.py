import subprocess
import sys
import traceback
from pathlib import Path

import pytest

from applications.ColossalChat.coati.distributed.reward.code_reward.testing_util import _load_runtime_module

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_NAME = "_colossalai_test_candidate"


@pytest.fixture(autouse=True)
def cleanup_runtime_module():
    yield
    sys.modules.pop(MODULE_NAME, None)


def _run_isolated(script: str):
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_load_runtime_module_supports_functions_and_solution_classes():
    module = _load_runtime_module(
        MODULE_NAME,
        "def add(a, b):\n    return a + b\n\nclass Solution:\n    def double(self, value):\n        return value * 2\n",
    )

    assert module.__name__ == MODULE_NAME
    assert module.add(2, 3) == 5
    assert module.Solution().double(4) == 8


def test_load_runtime_module_does_not_share_namespaces():
    first = _load_runtime_module(MODULE_NAME, "value = 1")
    second = _load_runtime_module(MODULE_NAME, "value = 2")

    assert first is not second
    assert first.value == 1
    assert second.value == 2
    assert sys.modules[MODULE_NAME] is second


def test_load_runtime_module_supports_dataclasses():
    module = _load_runtime_module(
        MODULE_NAME,
        "from dataclasses import dataclass\n\n@dataclass\nclass Point:\n    x: int\n    y: int\n",
    )

    assert module.Point(2, 3) == module.Point(x=2, y=3)


def test_load_runtime_module_uses_string_filename_in_tracebacks():
    previous = _load_runtime_module(MODULE_NAME, "value = 1")

    with pytest.raises(RuntimeError) as exc_info:
        _load_runtime_module(MODULE_NAME, "raise RuntimeError('boom')")

    frames = traceback.extract_tb(exc_info.value.__traceback__)
    assert frames[-1].filename == "<string>"
    assert sys.modules[MODULE_NAME] is previous


def test_load_runtime_module_reports_syntax_errors():
    with pytest.raises(SyntaxError) as exc_info:
        _load_runtime_module(MODULE_NAME, "def broken(:\n    pass")

    assert exc_info.value.filename == "<string>"


@pytest.mark.skipif(sys.platform == "win32", reason="The code verifier uses Unix process controls")
def test_run_test_executes_call_based_code_without_pyext():
    _run_isolated("""
from applications.ColossalChat.coati.distributed.reward.code_reward.testing_util import run_test

result, metadata = run_test(
    {"fn_name": "add", "inputs": ["2\\n3"], "outputs": ["5"]},
    test="def add(a, b):\\n    return a + b",
)
assert result == [True], (result, metadata)
""")


@pytest.mark.skipif(sys.platform == "win32", reason="The code verifier uses Unix process controls")
def test_run_test_executes_standard_input_code_without_pyext():
    _run_isolated("""
from applications.ColossalChat.coati.distributed.reward.code_reward.testing_util import run_test

result, metadata = run_test(
    {"inputs": ["4"], "outputs": ["8"]},
    test="value = int(input())\\nprint(value * 2)",
)
assert result == [True], (result, metadata)
""")
