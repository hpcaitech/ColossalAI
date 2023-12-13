from typing import Union

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from colossalai.logging import get_dist_logger

logger = get_dist_logger(__name__)


def get_tokenizer(
    tokenizer_name: str,
    use_fast_tokenizer: bool = False,
    trust_remote_code: bool = False,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Get Tokenizer

    Args:
        tokenizer_name (str): Name or path of the huggingface tokenizer
        use_fast_tokenizer (bool, optional): Whether to use fast tokenizer. Defaults to False.
        trust_remote_code (bool, optional): Whether to trust remote code from huggingface. Defaults to False.

    Returns:
        Union[PreTrainedTokenizer, PreTrainedTokenizerFast]: The obtained tokenizer.
    """

    return AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=trust_remote_code, use_fast=use_fast_tokenizer, padding_side="left"
    )
