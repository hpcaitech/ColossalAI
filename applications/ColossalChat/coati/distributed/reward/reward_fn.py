import torch
from latex2sympy2_extended import NormalizationConfig
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

from .reward_utils import extract_boxed_solution, extract_solution, validate_response_structure

CANNOT_PARSE_GT_ANSWER = -1
CANNOT_PARSE_PREDICTION = -2
SUCCESS = 1
MATCHING_FAIL = 0


def verify_math_representation(completion, gt_answer):
    """
    Verify if the completion is a valid math representation of the gt_answer.
    """
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
    if math_verify_result == SUCCESS:
        ans_acc += 1
        reward += acc_score
    elif math_verify_result == CANNOT_PARSE_GT_ANSWER or math_verify_result == CANNOT_PARSE_PREDICTION:
        if decoded_final_answer.strip().replace(" ", "").replace("{", "").replace("}", "").replace(
            ",", ""
        ) == gt_answer.strip().replace(" ", "").replace("{", "").replace("}", "").replace(",", ""):
            ans_acc += 1
            if math_verify_result == CANNOT_PARSE_GT_ANSWER:
                # plain text answer cannot be parsed, but is correct
                reward += acc_score
            else:
                reward += (
                    acc_score / 2
                )  # not a valid latex math representation, but the answer is correct, receive half of the score
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
    if soft_over_length_punishment:
        max_length = kwargs.get("max_length", 1024 * 4)
        cache_length = kwargs.get("cache_length", 512)
        res_length = e.item() - s.item() + 1
        if max_length - cache_length < res_length < max_length:
            length_reward = ((max_length - cache_length) - res_length) / cache_length * acc_score

    if gt_answer is None:
        return reward

    decoded_final_answer = tokenizer.decode(input_ids[s : e + 1], skip_special_tokens=True)
    gt_answer = tokenizer.decode(gt_answer.squeeze(0), skip_special_tokens=True)
    final_answer, processed_str = extract_solution(decoded_final_answer)

    format_valid = validate_response_structure(processed_str, kwargs["tags"])

    # Check format accuracy
    if format_valid:
        format_acc += 1

    # Check answer accuracy, answer is considered correct if the answer is correct and the format is valid
    if format_valid and final_answer is not None:
        reward, ans_acc = verify_model_answer(decoded_final_answer, gt_answer, ans_acc, acc_score, reward)

    reward = reward + length_reward

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
        }


def boxed_math_reward_fn(input_ids, gt_answer, response_idx, **kwargs):
    tokenizer = kwargs["tokenizer"]
    eval_mode = kwargs.get("eval_mode", False)
    soft_over_length_punishment = kwargs.get("soft_over_length_punishment", False)
    format_score = 0.0
    acc_score = 10.0
    reward = torch.tensor(0.0)
    format_acc = torch.tensor(0.0)
    ans_acc = torch.tensor(0.0)
    s, e = response_idx[0], response_idx[1]

    length_reward = 0.0
    if soft_over_length_punishment:
        max_length = kwargs.get("max_length", 1024 * 4)
        cache_length = kwargs.get("cache_length", 512)
        res_length = e.item() - s.item() + 1
        if max_length - cache_length < res_length < max_length:
            length_reward = ((max_length - cache_length) - res_length) / cache_length * acc_score

    if gt_answer is None:
        return torch.tensor([reward, format_acc, ans_acc]).to(input_ids.device)

    decoded_final_answer = tokenizer.decode(input_ids[s : e + 1], skip_special_tokens=True)
    gt_answer = tokenizer.decode(gt_answer.squeeze(0), skip_special_tokens=True)
    final_answer = extract_boxed_solution(decoded_final_answer)
    format_valid = final_answer is not None
    # Check format accuracy
    if format_valid:
        format_acc += 1
        reward += format_score

    # Check answer accuracy, answer is considered correct if the answer is correct and the format is valid
    if format_valid and final_answer is not None:
        reward, ans_acc = verify_model_answer(decoded_final_answer, gt_answer, ans_acc, acc_score, reward)

    reward = reward + length_reward
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
        }
