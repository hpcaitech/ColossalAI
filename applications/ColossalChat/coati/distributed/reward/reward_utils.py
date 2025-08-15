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


def extract_boxed_solution(text: str) -> Optional[str]:
    """
    Modified from: https://gist.github.com/lewtun/9c2ce1937b741404090a3dc4c7c022b3
    Retrieves the content from the last occurrence of `\boxed{}` in a LaTeX-like string.

    Args:
        text (str): A string potentially containing LaTeX-style boxed expressions.

    Returns:
        Optional[str]: The text inside the final `\boxed{}` if successfully extracted;
                       returns `None` if no properly closed box is found.

    Examples:
        >>> extract_boxed_solution("The answer is \\boxed{42}.")
        '42'
        >>> extract_boxed_solution("Here is an unmatched \\boxed{42")
        None
    """
    try:
        # Find the last occurrence of "\boxed{"
        start_idx = text.rindex("\\boxed{")
        # Move past "\boxed{" to find the start of the content
        content_start = start_idx + len("\\boxed{")
        open_braces = 1
        pos = content_start

        # Traverse the string to find the matching closing brace
        while open_braces > 0 and pos < len(text):
            if text[pos] == "{":
                open_braces += 1
            elif text[pos] == "}":
                open_braces -= 1
            pos += 1

        # If all braces are matched, extract and return the content
        if open_braces == 0:
            return text[content_start : pos - 1].strip()
        else:
            return None

    except ValueError:
        # "\boxed{" not found
        return None
    except Exception:
        # Any other unexpected error
        return None
