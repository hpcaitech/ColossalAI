from typing import List, Optional, Tuple

import torch
from transformers.generation import GenerationConfig

from colossalai.inference.logit_processors import get_logits_processor


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


def _sample(probs: torch.Tensor, logprobs: torch.Tensor, generation_config: GenerationConfig, is_prompt: bool = False):
    if generation_config.num_beams == 1:
        if generation_config.do_sample:
            sample_tokens = multinomial_sample(generation_config, probs)
        else:
            sample_tokens = greedy_sample(generation_config, logprobs)
    else:
        sample_tokens = beam_search_sample(generation_config, logprobs, is_prompt=is_prompt)

    return sample_tokens


def search_tokens(
    generation_config: GenerationConfig,
    logits,
    is_prompt: bool = False,
    batch_token_ids: Optional[List[List[int]]] = None,
):
    """
    Sample tokens for finished requests.
    """
    # NOTE: need to decide the granularity to process logits (sequence or batch)
    print(
        f"CHECK search_tokens max_length {generation_config.max_length}; max_new_tokens {generation_config.max_new_tokens}"
    )
    config_dict = generation_config.to_dict()
    if (repetition_penalty := config_dict.get("repetition_penalty", 1.0)) != 1.0:
        logits = get_logits_processor("repetition_penalty", logits, repetition_penalty, batch_token_ids)
    if (no_repeat_ngram_size := config_dict.get("no_repeat_ngram_size", 0)) > 0:
        logits = get_logits_processor("no_repeat_ngram_size", logits, no_repeat_ngram_size, batch_token_ids)
    if (forced_eos_token_id := config_dict.get("forced_eos_token_id", None)) is not None:
        sequence_lengths = [len(batch_token_ids[i]) for i in range(len(batch_token_ids))]
        max_out_lengths = [generation_config.max_length for _ in range(len(batch_token_ids))]
        logits = get_logits_processor(
            "forced_eos_token_id", logits, sequence_lengths, max_out_lengths, forced_eos_token_id
        )

    if generation_config.do_sample:
        if (temperature := config_dict.get("temperature", 1.0)) != 1.0:
            logits = get_logits_processor("temperature", logits, temperature)
        if (top_k := config_dict.get("top_k", 0)) != 0:
            logits = get_logits_processor("top_k", logits, top_k)
        if (top_p := config_dict.get("top_p", 1.0)) < 1.0:
            logits = get_logits_processor("top_p", logits, top_p)

    # calculate probs
    probs = torch.softmax(logits, dim=-1, dtype=torch.float)
    logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

    # sample the next tokens
    sample_tokens = _sample(probs, logprobs, generation_config, is_prompt)
    return sample_tokens
