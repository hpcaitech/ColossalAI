"""Record CPU CI stacks in the parent and every spawned Python worker.

Loaded through the CPU gate's PYTHONPATH, preserving the existing python -m
pytest invocation and multiprocessing entry point. The gate uploads these logs.
"""

import atexit
import faulthandler
import os
import time
from pathlib import Path

_TRACE_STREAM = None
if os.environ.get("COLOSSAL_CI_TRACEBACK_DIR"):
    directory = Path(os.environ["COLOSSAL_CI_TRACEBACK_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    _TRACE_STREAM = (directory / f"pid-{os.getpid()}.log").open("a", buffering=1)
    print(
        f"START time={time.time():.3f} pid={os.getpid()} ppid={os.getppid()} "
        f"test={os.environ.get('PYTEST_CURRENT_TEST', '<startup>')}",
        file=_TRACE_STREAM,
        flush=True,
    )
    faulthandler.enable(file=_TRACE_STREAM, all_threads=True)
    faulthandler.dump_traceback_later(
        float(os.environ.get("COLOSSAL_CI_TRACEBACK_SECONDS", "60")),
        repeat=True,
        file=_TRACE_STREAM,
    )

    def _finish_tracebacks():
        faulthandler.cancel_dump_traceback_later()
        faulthandler.disable()
        print(f"EXIT time={time.time():.3f} pid={os.getpid()}", file=_TRACE_STREAM, flush=True)
        _TRACE_STREAM.close()

    atexit.register(_finish_tracebacks)
