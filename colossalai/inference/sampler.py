from typing import List, Optional, Tuple, Union

import torch
from transformers.generation import GenerationConfig

from colossalai.inference.logit_processors import get_logits_processor


def greedy_sample(
    logprobs: torch.Tensor,
) -> torch.Tensor:
    """
    Sample tokens greedyly.
    """
    results = torch.argmax(logprobs, dim=-1)
    return results


def multinomial_sample(
    probs: torch.Tensor,
) -> torch.Tensor:
    """
    Sample tokens in a random phase.
    """
    random_results = torch.multinomial(probs, num_samples=1).squeeze(1)
    return random_results


def beam_search_sample(
    beam_width: int,
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


def search_tokens(
    generation_config: Union[GenerationConfig, dict],
    logits,
    is_prompt: bool = False,
    batch_token_ids: Optional[List[List[int]]] = None,
):
    """
    Sample tokens for finished requests.
    """
    # NOTE: need to decide the granularity to process logits (sequence or batch)

    # convert GenerationConfig to dict
    # temporary fix for compatibility with the usage of RPCInferenceEngine
    if isinstance(generation_config, GenerationConfig):
        generation_config = generation_config.to_dict()

    if (repetition_penalty := generation_config.get("repetition_penalty", 1.0)) != 1.0:
        logits = get_logits_processor("repetition_penalty", logits, repetition_penalty, batch_token_ids)
    if (no_repeat_ngram_size := generation_config.get("no_repeat_ngram_size", 0)) > 0:
        logits = get_logits_processor("no_repeat_ngram_size", logits, no_repeat_ngram_size, batch_token_ids)
    if (forced_eos_token_id := generation_config.get("forced_eos_token_id", None)) is not None:
        sequence_lengths = [len(batch_token_ids[i]) for i in range(len(batch_token_ids))]
        max_out_lengths = [generation_config.max_length for _ in range(len(batch_token_ids))]
        logits = get_logits_processor(
            "forced_eos_token_id", logits, sequence_lengths, max_out_lengths, forced_eos_token_id
        )

    if generation_config.get("do_sample"):
        if (temperature := generation_config.get("temperature", 1.0)) != 1.0:
            logits = get_logits_processor("temperature", logits, temperature)
        if (top_k := generation_config.get("top_k", 0)) != 0:
            logits = get_logits_processor("top_k", logits, top_k)
        if (top_p := generation_config.get("top_p", 1.0)) < 1.0:
            logits = get_logits_processor("top_p", logits, top_p)

    # calculate probs
    probs = torch.softmax(logits, dim=-1, dtype=torch.float)
    logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

    # sample the next tokens
    if generation_config.get("num_beams", 1) != 1:
        raise NotImplementedError("Beam search is not supported yet.")
    if generation_config.get("do_sample", False):
        sample_tokens = multinomial_sample(probs)
    else:
        sample_tokens = greedy_sample(logprobs)

    return sample_tokens
