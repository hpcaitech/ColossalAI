from typing import List, Tuple

import torch


def greedy_sample(
    generation_config,
    logprobs: torch.Tensor,
) -> torch.Tensor:
    """
    Sample tokens greedyly.
    """
    results = torch.argmax(logprobs, dim=-1)
    return results


def multinomial_sample(
    generation_config,
    probs: torch.Tensor,
) -> torch.Tensor:
    """
    Sample tokens in a random phase.
    """
    random_results = torch.multinomial(probs, num_samples=1).squeeze(1)
    return random_results


def beam_search_sample(
    generation_config,
    logprobs: torch.Tensor,
    is_prompt: bool = False,
) -> List[Tuple[List[int], List[int]]]:
    """
    Sample tokens with beam search.
    We sample 2 * beam_width candidates to make sure that with high probability we can get `beam_width` candidates in addition to
    the finished sequences for the next iteration.

    ref:
        https://github.com/tensorflow/tensor2tensor/blob/bafdc1b67730430d38d6ab802cbd51f9d053ba2e/tensor2tensor/utils/beam_search.py#L557-L563
    for details. See also HF reference:
        https://github.com/huggingface/transformers/blob/a4dd53d88e4852f023332d284ff07a01afcd5681/src/transformers/generation/utils.py#L3063-L3065

    # NOTE: this beam search sample function is wrong now.
    """

    beam_width = generation_config.num_beams
    results = []
    if is_prompt:
        # Prompt phase.
        parent_ids = [0] * (2 * beam_width)
        _, next_token_ids = torch.topk(logprobs[0], 2 * beam_width)
        next_token_ids = next_token_ids.tolist()
    else:
        # Generation phase.
        # cumulative_logprobs = [seq_data[seq_id].cumulative_logprob for seq_id in seq_ids]
        cumulative_logprobs = torch.tensor(logprobs, dtype=torch.float, device=seq_group_logprobs.device)
        seq_group_logprobs = seq_group_logprobs + cumulative_logprobs.unsqueeze(dim=1)
        _, topk_ids = torch.topk(logprobs.flatten(), 2 * beam_width)

    results.append((next_token_ids, parent_ids))
    return results
