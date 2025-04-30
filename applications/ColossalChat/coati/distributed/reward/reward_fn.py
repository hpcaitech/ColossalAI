import torch

from .reward_utils import extract_boxed_solution, extract_solution, validate_response_structure


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
    if (
        format_valid
        and final_answer is not None
        and gt_answer.strip().replace(" ", "").lower() == final_answer.strip().replace(" ", "").lower()
    ):
        ans_acc += 1
        reward += acc_score

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
    if format_valid and final_answer is not None and gt_answer.strip().lower() == final_answer.strip().lower():
        ans_acc += 1
        reward += acc_score

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
