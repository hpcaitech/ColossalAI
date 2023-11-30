"""
Motivated by VllM (https://github.com/vllm-project/vllm), This module is trying to resolve the tokenizer issue.

license: MIT, see LICENSE for more details.
"""

from transformers import AutoTokenizer

_FAST_LLAMA_TOKENIZER = "hf-internal-testing/llama-tokenizer"


def get_tokenizer(
    tokenizer=None,
    tokenizer_name: str = "",
    trust_remote_code: bool = False,
    use_fast: bool = True,
):
    if tokenizer is not None:
        tokenizer = tokenizer
    else:
        if "llama" in tokenizer_name.lower() and use_fast == True:
            print(
                "For some LLaMA-based models, initializing the fast tokenizer may "
                "take a long time. To eliminate the initialization time, consider "
                f"using '{_FAST_LLAMA_TOKENIZER}' instead of the original "
                "tokenizer. This is done automatically in Colossalai."
            )

            tokenizer_name = _FAST_LLAMA_TOKENIZER

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, use_fast=use_fast, trust_remote_code=trust_remote_code
            )
        except TypeError:
            use_fast = False
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, use_fast=use_fast, trust_remote_code=trust_remote_code
            )
    return tokenizer
