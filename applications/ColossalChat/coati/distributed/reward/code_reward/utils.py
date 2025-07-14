# Code from the verl Project (https://github.com/agentica-project/rllm),
# which itself is adapted from Prime (https://github.com/PRIME-RL/PRIME)
#
# Copyright 2024 ByteDance Group

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing
import traceback
from typing import Optional

import requests

from .testing_util import run_test


def _temp_run(sample, generation, debug, result, metadata_list, timeout):
    try:
        res, metadata = run_test(in_outs=sample, test=generation, debug=debug, timeout=timeout)
        result.append(res)
        metadata_list.append(metadata)
    except Exception:
        # print(e) # some tracebacks are extremely long.
        traceback.print_exc(10)
        result.append([-1 for i in range(len(sample["inputs"]))])
        metadata_list.append({})


def check_correctness(in_outs: Optional[dict], generation, timeout=10, debug=True):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""

    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(in_outs, generation, debug, result, metadata_list, timeout))
    p.start()
    p.join(timeout=600)  # Global timeout of 10 minutes that's for all test cases combined
    if p.is_alive():
        p.kill()
        # p.terminate()
    if not result:
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        if debug:
            print("global timeout")
    return result[0], metadata_list


def check_correctness_code_api(
    in_outs: Optional[dict], generation, timeout=10, debug=True, url="http://localhost:8000/check_correctness"
):
    payload = {"in_outs": in_outs, "generation": generation, "timeout": timeout, "debug": debug}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        results = response.json()
        return results["result"], results["metadata"]
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return [-1 for i in range(len(in_outs["inputs"]))], {}
