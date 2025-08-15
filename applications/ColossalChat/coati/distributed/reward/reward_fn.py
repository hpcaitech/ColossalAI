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
"""
Some functions in this file are adapted from the verl project
under the Apache License 2.0:
https://github.com/volcengine/verl
"""


import json

import torch
from latex2sympy2_extended import NormalizationConfig
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

from .code_reward.utils import check_correctness_code_api as check_correctness_code
from .reward_utils import extract_boxed_solution, extract_solution, validate_response_structure

CANNOT_PARSE_GT_ANSWER = -1
CANNOT_PARSE_PREDICTION = -2
SUCCESS = 1
MATCHING_FAIL = 0


def verify_math_representation(completion, gt_answer):
    """
    Verify if the completion is a valid math representation of the gt_answer.
    """
    if not completion.startswith("\\boxed{"):
        completion = "\\boxed{" + completion + "}"
    if not gt_answer.startswith("\\boxed{"):
        gt_answer = "\\boxed{" + gt_answer + "}"
    target = (
        ExprExtractionConfig(),
        LatexExtractionConfig(
            normalization_config=NormalizationConfig(
                nits=False,
                malformed_operators=False,
                basic_latex=True,
                boxed="all",
                units=True,
            ),
            boxed_match_priority=0,
        ),
    )
    if not isinstance(gt_answer, str) or len(gt_answer) == 0:
        raise ValueError("gt_answer should be a string, please verify your training data.")
    if not isinstance(completion, str) or len(completion) == 0:
        return MATCHING_FAIL
    try:
        parsed_gt_answer = parse(gt_answer, extraction_config=target)
        if len(parsed_gt_answer) == 0:
            return CANNOT_PARSE_GT_ANSWER
        parsed_completion = parse(completion, extraction_config=target)
        if len(parsed_completion) == 0:
            return CANNOT_PARSE_PREDICTION
        if verify(parsed_gt_answer, parsed_completion):
            return SUCCESS
        else:
            return MATCHING_FAIL
    except Exception:
        return MATCHING_FAIL


def verify_model_answer(decoded_final_answer, gt_answer, ans_acc, acc_score, reward):
    math_verify_result = verify_math_representation(decoded_final_answer, gt_answer)
    exact_match_result = (
        SUCCESS
        if decoded_final_answer.strip().replace(" ", "").replace("{", "").replace("}", "").replace(",", "")
        == gt_answer.strip().replace(" ", "").replace("{", "").replace("}", "").replace(",", "")
        else MATCHING_FAIL
    )
    if math_verify_result == SUCCESS:
        ans_acc += 1
        reward += acc_score
    elif exact_match_result == SUCCESS:
        # sometimes for answers that's not a (valid) math expression, math_verify will fail
        ans_acc += 1
        if math_verify_result == CANNOT_PARSE_PREDICTION:
            reward += (
                acc_score / 2
            )  # not a valid latex math representation, but the answer is correct, receive half of the score
        else:
            reward += acc_score
    return reward, ans_acc


def math_reward_fn(input_ids, gt_answer, response_idx, **kwargs):
    tokenizer = kwargs["tokenizer"]
    eval_mode = kwargs.get("eval_mode", False)
    soft_over_length_punishment = kwargs.get("soft_over_length_punishment", False)
    acc_score = 10.0
    reward = torch.tensor(0.0)
    format_acc = torch.tensor(0.0)
    ans_acc = torch.tensor(0.0)
    s, e = response_idx[0], response_idx[1]

    length_reward = 0.0
    res_length = e.item() - s.item() + 1
    if not eval_mode:
        max_new_tokens = kwargs["max_new_tokens"]
    else:
        max_new_tokens = -1  # for eval mode, we don't need to check the length
    if not eval_mode and soft_over_length_punishment:
        cache_length = kwargs["cache_length"]
        if max_new_tokens - cache_length < res_length < max_new_tokens:
            length_reward = ((max_new_tokens - cache_length) - res_length) / cache_length * acc_score

    if gt_answer is None:
        raise ValueError("no gt_answer is provided, please check your training dataset.")

    decoded_final_answer = tokenizer.decode(input_ids[s : e + 1], skip_special_tokens=True)

    final_answer, processed_str = extract_solution(decoded_final_answer)

    format_valid = validate_response_structure(processed_str, kwargs["tags"])

    # Check answer accuracy, answer is considered correct if the answer is correct and the format is valid
    if final_answer is not None:
        if eval_mode or format_valid:
            reward, ans_acc = verify_model_answer(final_answer, gt_answer, ans_acc, acc_score, reward)
        if not eval_mode:
            reward = reward + length_reward

    # Check format accuracy
    if format_valid:
        format_acc += 1

    # Check if the sequence is over length
    if not eval_mode and res_length >= max_new_tokens:
        reward *= 0.0

    if not eval_mode:
        return torch.tensor([reward, format_acc, ans_acc]).to(input_ids.device)
    else:
        prompt = tokenizer.decode(input_ids[:s], skip_special_tokens=True)
        return {
            "prompt": prompt,
            "prediction": decoded_final_answer,
            "gold": gt_answer,
            "parsed": final_answer,
            "format_valid": format_acc.item(),
            "ans_valid": ans_acc.item(),
            "response_length": res_length,
            "reward": reward.item(),
        }


def boxed_math_reward_fn(input_ids, gt_answer, response_idx, **kwargs):
    tokenizer = kwargs["tokenizer"]
    eval_mode = kwargs.get("eval_mode", False)
    soft_over_length_punishment = kwargs.get("soft_over_length_punishment", False)
    acc_score = 10.0
    reward = torch.tensor(0.0)
    format_acc = torch.tensor(0.0)
    ans_acc = torch.tensor(0.0)
    s, e = response_idx[0], response_idx[1]

    length_reward = 0.0
    res_length = e.item() - s.item() + 1
    if not eval_mode:
        max_new_tokens = kwargs["max_new_tokens"]
    else:
        max_new_tokens = -1  # for eval mode, we don't need to check the length
    if not eval_mode and soft_over_length_punishment:
        cache_length = kwargs["cache_length"]
        if max_new_tokens - cache_length < res_length < max_new_tokens:
            length_reward = ((max_new_tokens - cache_length) - res_length) / cache_length * acc_score

    if gt_answer is None:
        raise ValueError("no gt_answer is provided, please check your training dataset.")

    decoded_final_answer = tokenizer.decode(input_ids[s : e + 1], skip_special_tokens=True)

    final_answer = extract_boxed_solution(decoded_final_answer)
    format_valid = final_answer is not None
    if "tags" in kwargs and kwargs["tags"]:
        tags = kwargs["tags"]
        format_valid = format_valid and all(
            [decoded_final_answer.count(tags[tag]["text"]) == tags[tag]["num_occur"] for tag in tags]
        )

    # Check answer accuracy, answer is considered correct if the answer is correct and the format is valid
    if final_answer is not None:
        if eval_mode or format_valid:
            reward, ans_acc = verify_model_answer(final_answer, gt_answer, ans_acc, acc_score, reward)
        if not eval_mode:
            reward = reward + length_reward

    # Check format accuracy
    if format_valid:
        format_acc += 1

    # Check if the sequence is over length
    if not eval_mode and res_length >= max_new_tokens:
        reward *= 0.0

    if not eval_mode:
        return torch.tensor([reward, format_acc, ans_acc]).to(input_ids.device)
    else:
        prompt = tokenizer.decode(input_ids[:s], skip_special_tokens=True)
        return {
            "prompt": prompt,
            "prediction": decoded_final_answer,
            "gold": gt_answer,
            "parsed": final_answer,
            "format_valid": format_acc.item(),
            "ans_valid": ans_acc.item(),
            "response_length": res_length,
            "reward": reward.item(),
        }


def code_reward_fn(input_ids, test_cases, response_idx, **kwargs):
    url = kwargs.get("url", "http://localhost:8000/check_correctness")
    tokenizer = kwargs["tokenizer"]
    eval_mode = kwargs.get("eval_mode", False)
    soft_over_length_punishment = kwargs.get("soft_over_length_punishment", False)
    acc_score = 10.0
    reward = torch.tensor(0.0)
    format_acc = torch.tensor(0.0)
    ans_acc = torch.tensor(0.0)
    s, e = response_idx[0], response_idx[1]

    length_reward = 0.0
    res_length = e.item() - s.item() + 1
    if not eval_mode:
        max_new_tokens = kwargs["max_new_tokens"]
    else:
        max_new_tokens = -1  # for eval mode, we don't need to check the length
    if not eval_mode and soft_over_length_punishment:
        cache_length = kwargs["cache_length"]
        if max_new_tokens - cache_length < res_length < max_new_tokens:
            length_reward = ((max_new_tokens - cache_length) - res_length) / cache_length * acc_score

    # try to get code solution from completion. if the completion is pure code, this will not take effect.
    decoded_final_answer = tokenizer.decode(input_ids[s : e + 1], skip_special_tokens=True)

    solution = decoded_final_answer.split("```python")[-1].split("```")[0]
    format_valid = False
    if "```python" in decoded_final_answer:
        format_valid = solution is not None

    # Check format accuracy
    if format_valid:
        format_acc += 1

    res = []
    metadata = []

    try:
        try:
            if not isinstance(test_cases, dict):
                test_cases = json.loads(test_cases)
        except Exception as e:
            print(f"Error {e}: Cannot parse test cases.")
            raise e
        # Complete check on all in-out pairs first. If there is no failure, per-sample test can be skipped.
        try:
            res, metadata = check_correctness_code(
                in_outs=test_cases, generation=solution, timeout=10, debug=False, url=url
            )
            metadata = dict(enumerate(metadata))[0]
            success = all(map(lambda x: x == 1, res))
            if success:
                ans_acc += 1
                if eval_mode or format_valid:
                    reward += acc_score
                if not eval_mode:
                    reward = reward + length_reward

        except Exception:
            pass

        # Check if the sequence is over length
        if not eval_mode and res_length >= max_new_tokens:
            reward *= 0.0
    except Exception:
        pass
    if not eval_mode:
        return torch.tensor([reward, format_acc, ans_acc]).to(input_ids.device)
    else:
        prompt = tokenizer.decode(input_ids[:s], skip_special_tokens=True)
        return {
            "prompt": prompt,
            "prediction": decoded_final_answer,
            "test_cases": test_cases,
            "test_results": res,
            "test_metadata": metadata,
            "parsed": solution,
            "format_valid": format_acc.item(),
            "ans_valid": ans_acc.item(),
            "response_length": res_length,
            "reward": reward.item(),
        }
