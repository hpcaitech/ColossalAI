import torch

from .reward_utils import extract_solution, validate_response_structure


def math_reward_fn(input_ids, gt_answer, response_idx, **kwargs):
    tokenizer = kwargs["tokenizer"]
    reward = torch.tensor(0.0).to(input_ids.device)
    s, e = response_idx[0], response_idx[1]
    if gt_answer is None:
        return reward

    decoded_final_answer = tokenizer.decode(input_ids[s : e + 1], skip_special_tokens=True)
    gt_answer = tokenizer.decode(gt_answer.squeeze(0), skip_special_tokens=True)
    final_answer, processed_str = extract_solution(decoded_final_answer)

    format_valid = validate_response_structure(processed_str, kwargs["tags"])
    if not format_valid:
        return reward
    else:
        reward += 1.0
        # if gt_answer.strip().replace(" ", "").lower() == final_answer.strip().replace(" ", "").lower():
        #     reward = reward + 2.0
        return reward


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
