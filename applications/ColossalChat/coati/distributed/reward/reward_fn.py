import torch

from .reward_utils import extract_solution, validate_response_structure


def math_reward_fn(input_ids, gt_answer, response_idx, **kwargs):
    tokenizer = kwargs["tokenizer"]
    soft_over_length_punishment = kwargs["soft_over_length_punishment"]
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
        return reward

    decoded_final_answer = tokenizer.decode(input_ids[s : e + 1], skip_special_tokens=True)
    gt_answer = tokenizer.decode(gt_answer.squeeze(0), skip_special_tokens=True)
    final_answer, processed_str = extract_solution(decoded_final_answer)

    format_valid = validate_response_structure(processed_str, kwargs["tags"])

    # Check format accuracy
    if format_valid:
        format_acc += 1
        reward += format_score

    # Check answer accuracy
    if (
        final_answer is not None
        and gt_answer.strip().replace(" ", "").lower() == final_answer.strip().replace(" ", "").lower()
    ):
        ans_acc += 1
        reward += acc_score

    reward = reward + length_reward

    return torch.tensor([reward, format_acc, ans_acc]).to(input_ids.device)


def gsm8k_reward_fn(input_ids, **kwargs):
    gt_answer = kwargs["gt_answer"]
    tokenizer = kwargs["tokenizer"]
    s, e = kwargs["response_start"], kwargs["response_end"]
    reward = torch.tensor(0.0).to(input_ids.device)
    if gt_answer is None:
        return reward
    decoded_final_answer = tokenizer.decode(input_ids[s : e + 1], skip_special_tokens=True)
    final_answer, processed_str = extract_solution(decoded_final_answer)
    is_valid = True
    try:
        int(final_answer.strip())
    except Exception:
        is_valid = False

    format_valid = validate_response_structure(processed_str, kwargs["tags"])
    if not is_valid or not format_valid:
        return reward
    else:
        reward += 1.0
        if gt_answer.strip().replace(" ", "").lower() == final_answer.strip().replace(" ", "").lower():
            reward = reward + 9.0
        return reward
