from functools import lru_cache

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


@lru_cache(maxsize=None)
def get_test_tokenizer(vocab_size: int = 32000) -> PreTrainedTokenizerFast:
    """Build a deterministic tokenizer for tests without accessing the Hub."""
    if vocab_size < 3:
        raise ValueError("vocab_size must leave room for the test special tokens")

    vocab = {"<unk>": 0, "<pad>": 1, "</s>": 2}
    vocab.update({f"token_{index}": index for index in range(3, vocab_size)})
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="</s>",
    )
