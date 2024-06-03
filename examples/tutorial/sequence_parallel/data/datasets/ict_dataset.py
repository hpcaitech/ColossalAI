import itertools
import random

import numpy as np
from megatron import get_args, get_tokenizer
from megatron.data.dataset_utils import get_indexed_dataset_
from megatron.data.realm_dataset_utils import get_block_samples_mapping
from torch.utils.data import Dataset


def make_attention_mask(source_block, target_block):
    """
    Returns a 2-dimensional (2-D) attention mask
    :param source_block: 1-D array
    :param target_block: 1-D array
    """
    mask = (target_block[None, :] >= 1) * (source_block[:, None] >= 1)
    mask = mask.astype(np.int64)
    # (source_length, target_length)
    return mask


def get_ict_dataset(use_titles=True, query_in_block_prob=1):
    """Get a dataset which uses block samples mappings to get ICT/block indexing data (via get_block())
    rather than for training, since it is only built with a single epoch sample mapping.
    """
    args = get_args()
    block_dataset = get_indexed_dataset_(args.data_path, "mmap", True)
    titles_dataset = get_indexed_dataset_(args.titles_data_path, "mmap", True)

    kwargs = dict(
        name="full",
        block_dataset=block_dataset,
        title_dataset=titles_dataset,
        data_prefix=args.data_path,
        num_epochs=1,
        max_num_samples=None,
        max_seq_length=args.seq_length,
        seed=1,
        query_in_block_prob=query_in_block_prob,
        use_titles=use_titles,
        use_one_sent_docs=args.use_one_sent_docs,
    )
    dataset = ICTDataset(**kwargs)
    return dataset


class ICTDataset(Dataset):
    """Dataset containing sentences and their blocks for an inverse cloze task."""

    def __init__(
        self,
        name,
        block_dataset,
        title_dataset,
        data_prefix,
        num_epochs,
        max_num_samples,
        max_seq_length,
        query_in_block_prob,
        seed,
        use_titles=True,
        use_one_sent_docs=False,
        binary_head=False,
    ):
        self.name = name
        self.seed = seed
        self.max_seq_length = max_seq_length
        self.query_in_block_prob = query_in_block_prob
        self.block_dataset = block_dataset
        self.title_dataset = title_dataset
        self.rng = random.Random(self.seed)
        self.use_titles = use_titles
        self.use_one_sent_docs = use_one_sent_docs

        self.samples_mapping = get_block_samples_mapping(
            block_dataset,
            title_dataset,
            data_prefix,
            num_epochs,
            max_num_samples,
            max_seq_length,
            seed,
            name,
            use_one_sent_docs,
        )
        self.tokenizer = get_tokenizer()
        self.vocab_id_list = list(self.tokenizer.inv_vocab.keys())
        self.vocab_id_to_token_list = self.tokenizer.inv_vocab
        self.cls_id = self.tokenizer.cls
        self.sep_id = self.tokenizer.sep
        self.mask_id = self.tokenizer.mask
        self.pad_id = self.tokenizer.pad

    def __len__(self):
        return len(self.samples_mapping)

    def __getitem__(self, idx):
        """Get an ICT example of a pseudo-query and the block of text from which it was extracted"""
        sample_data = self.samples_mapping[idx]
        start_idx, end_idx, doc_idx, block_idx = sample_data.as_tuple()

        if self.use_titles:
            title = self.title_dataset[int(doc_idx)]
            title_pad_offset = 3 + len(title)
        else:
            title = None
            title_pad_offset = 2
        block = [self.block_dataset[i] for i in range(start_idx, end_idx)]
        assert len(block) > 1 or self.use_one_sent_docs or self.query_in_block_prob == 1

        # randint() is inclusive for Python rng
        rand_sent_idx = self.rng.randint(0, len(block) - 1)

        # keep the query in the context query_in_block_prob fraction of the time.
        if self.rng.random() < self.query_in_block_prob:
            query = block[rand_sent_idx].copy()
        else:
            query = block.pop(rand_sent_idx)

        # still need to truncate because blocks are concluded when
        # the sentence lengths have exceeded max_seq_length.
        query = query[: self.max_seq_length - 2]
        block = list(itertools.chain(*block))[: self.max_seq_length - title_pad_offset]

        query_tokens, query_pad_mask = self.concat_and_pad_tokens(query)
        context_tokens, context_pad_mask = self.concat_and_pad_tokens(block, title)

        query_mask = make_attention_mask(query_tokens, query_tokens)
        context_mask = make_attention_mask(context_tokens, context_tokens)

        block_data = sample_data.as_array()

        sample = {
            "query_tokens": query_tokens,
            "query_mask": query_mask,
            "query_pad_mask": query_pad_mask,
            "context_tokens": context_tokens,
            "context_mask": context_mask,
            "context_pad_mask": context_pad_mask,
            "block_data": block_data,
        }

        return sample

    def get_block(self, start_idx, end_idx, doc_idx):
        """Get the IDs for an evidence block plus the title of the corresponding document"""
        block = [self.block_dataset[i] for i in range(start_idx, end_idx)]
        title = self.title_dataset[int(doc_idx)]

        block = list(itertools.chain(*block))[: self.max_seq_length - (3 + len(title))]
        block_tokens, block_pad_mask = self.concat_and_pad_tokens(block, title)

        return block_tokens, block_pad_mask

    def get_null_block(self):
        """Get empty block and title - used in REALM pretraining"""
        block, title = [], []
        block_tokens, block_pad_mask = self.concat_and_pad_tokens(block, title)

        return block_tokens, block_pad_mask

    def concat_and_pad_tokens(self, tokens, title=None):
        """Concat with special tokens and pad sequence to self.max_seq_length"""
        tokens = list(tokens)
        if title is None:
            tokens = [self.cls_id] + tokens + [self.sep_id]
        else:
            title = list(title)
            tokens = [self.cls_id] + title + [self.sep_id] + tokens + [self.sep_id]
        assert len(tokens) <= self.max_seq_length

        num_pad = self.max_seq_length - len(tokens)
        pad_mask = [1] * len(tokens) + [0] * num_pad
        tokens += [self.pad_id] * num_pad

        return np.array(tokens), np.array(pad_mask)
