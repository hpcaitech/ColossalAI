# Copyright Unakar
# Modified from https://github.com/Unakar/Logic-RL/blob/086373176ac198c97277ff50f4b6e7e1bfe669d3/verl/utils/reward_score/kk.py#L99
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Dict, Optional, Tuple


def validate_response_structure(processed_str: str, tags: Dict = None) -> bool:
    """Performs comprehensive validation of response structure.

    Args:
        processed_str: Processed response string from the model

    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    validation_passed = True
    # Check required tags
    if tags is None:
        tags = {
            "think_start": {"text": "<think>", "num_occur": 1},
            "think_end": {"text": "</think>", "num_occur": 1},
            "answer_start": {"text": "<answer>", "num_occur": 1},
            "answer_end": {"text": "</answer>", "num_occur": 1},
        }
    positions = {}
    for tag_name, tag_info in tags.items():
        tag_str = tag_info["text"]
        expected_count = tag_info["num_occur"]
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        if count != expected_count:
            validation_passed = False
    # Verify tag order
    if (
        positions["think_start"] > positions["think_end"]
        or positions["think_end"] > positions["answer_start"]
        or positions["answer_start"] > positions["answer_end"]
    ):
        validation_passed = False
    if len(processed_str) - positions["answer_end"] != len(tags["answer_end"]["text"]):
        validation_passed = False
    return validation_passed


def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """Extracts the final answer from the model's response string.

    Args:
        solution_str: Raw response string from the language model

    Returns:
        Tuple containing (extracted_answer, processed_string)
    """

    # Extract final answer using XML-style tags
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))

    if not matches:
        return None, solution_str

    final_answer = matches[-1].group(1).strip()
    return final_answer, solution_str
