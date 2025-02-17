import torch

from .utils import extract_solution, validate_response_structure


def math_competition_reward_fn(input_ids, attention_mask, **kwargs):
    # apply varifiable reward
    # reward 10 points if the final answer is correct, reward 1 point if format is correct

    gt_answer = kwargs["gt_answer"]
    tokenizer = kwargs["tokenizer"]
    s, e = kwargs["response_start"], kwargs["response_end"]
    reward = torch.tensor(0.0).to(input_ids.device)
    if gt_answer is None:
        return reward
    decoded_final_answer = tokenizer.decode(input_ids[s : e + 1], skip_special_tokens=True)
    final_answer, processed_str = extract_solution(decoded_final_answer)

    format_valid = validate_response_structure(processed_str, kwargs["tags"])
    if not format_valid:
        return reward
    else:
        reward += 1.0
        if gt_answer.strip().replace(" ", "").lower() == final_answer.strip().replace(" ", "").lower():
            reward = reward + 9.0
        return reward
