import torch
from transformers.models.llama import LlamaConfig

from colossalai.inference.batch_bucket import BatchBucket
from colossalai.inference.config import InferenceConfig
from colossalai.inference.kv_cache import KVCacheManager
from colossalai.inference.struct import Sequence


def test_kvcache_manager():
    hidden_size = 128
    num_layers = 2
    num_heads = 4

    test_config = {
        "block_size": 4,
        "max_batch_size": 4,
        "max_input_len": 32,
        "max_output_len": 8,
        "dtype": torch.float16,
        "tp_size": 1,
    }

    inference_config = InferenceConfig(**test_config)
    model_config = LlamaConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
    )
    cache_manager = KVCacheManager(inference_config, model_config)

    context_lengths = torch.tensor([19, 20])
    block_tables = torch.tensor(
        [[-1 for _ in range(cache_manager.get_max_blocks_per_sequence())] for _ in range(2)], dtype=torch.int32
    )

    for i in range(2):
        cache_manager.allocate_context_from_block_table(block_tables[i], context_lengths[i])
    print(context_lengths)
    print(block_tables)
    print(cache_manager._cache_blocks[4])
    print(cache_manager._cache_blocks[9])

    context_lengths += 1

    cache_manager.allocate_tokens_from_block_tables(block_tables, context_lengths)
    print(context_lengths)
    print(block_tables)

    context_lengths += 1

    cache_manager.allocate_tokens_from_block_tables(block_tables, context_lengths)
    print(context_lengths)
    print(block_tables)
    context_lengths += 1


def test_bucket():
    hidden_size = 128
    num_layers = 2
    num_heads = 4

    test_config = {
        "block_size": 4,
        "max_batch_size": 4,
        "max_input_len": 32,
        "max_output_len": 8,
        "dtype": torch.float16,
        "tp_size": 1,
    }

    inference_config = InferenceConfig(**test_config)
    model_config = LlamaConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
    )
    cache_manager = KVCacheManager(inference_config, model_config)

    seq_lens = [19, 20, 27]

    seq1 = Sequence(
        request_id=0,
        prompt="",
        input_token_id=list(range(seq_lens[0])),
        block_size=4,
        sample_params=None,
        block_table=None,
        eos_token_id=2,
        pad_token_id=2,
        max_output_len=10,
    )
    seq2 = Sequence(
        request_id=1,
        prompt="bcd",
        input_token_id=list(range(seq_lens[1])),
        block_size=4,
        sample_params=None,
        block_table=None,
        eos_token_id=2,
        pad_token_id=2,
        max_output_len=10,
    )
    seq3 = Sequence(
        request_id=2,
        prompt="bcd",
        input_token_id=list(range(seq_lens[2])),
        block_size=4,
        sample_params=None,
        block_table=None,
        eos_token_id=2,
        pad_token_id=2,
        max_output_len=10,
    )

    bb = BatchBucket(num_heads, cache_manager.get_head_size(), 4, 32, 4, kv_max_split_num=2)
    # bb.init_batch_from_list([seq1, seq2])
    block_tables = bb.add_seqs([seq1, seq2])
    print("> After add_seqs")
    print(block_tables)
    cache_manager.allocate_context_from_block_tables(block_tables, bb.seq_lengths[: bb.current_batch_size])

    print("> After allocate_context_from_block_tables")
    print(bb.seq_lengths)
    print(bb.block_tables)

    bb.seq_lengths[: bb.current_batch_size] += 1

    cache_manager.allocate_tokens_from_block_tables(bb.block_tables, bb.seq_lengths, bsz=bb.current_batch_size)
    print("> After allocate_tokens_from_block_tables")
    print(bb.seq_lengths)
    print(bb.block_tables)

    bb.seq_lengths[: bb.current_batch_size] += 1

    cache_manager.allocate_tokens_from_block_tables(bb.block_tables, bb.seq_lengths, bsz=bb.current_batch_size)
    print("> After allocate_tokens_from_block_tables")
    print(bb.seq_lengths)
    print(bb.block_tables)
    # print(cache_manager._cache_blocks)

    bb.pop_seq_update_batch(0, cache_manager.free_block_table)
    print("> After pop_seq_update_batch")
    print(bb.seq_lengths)
    print(bb.block_tables)
    # print(cache_manager._cache_blocks)
    print("bb.current_batch_size: ", bb.current_batch_size)
    print(bb._sequences_indexes)
    print(bb._sequences_dict.keys())

    # bb.pop_seq_update_batch(1, cache_manager.free_block_table)
    # print("> After pop_seq_update_batch")
    # print(bb.seq_lengths)
    # print(bb.block_tables)
    # print(cache_manager._cache_blocks)
    # print("bb.current_batch_size: ", bb.current_batch_size)
    # print(bb._sequences_indexes)
    # print(bb._sequences_dict.keys())

    bb2 = BatchBucket(num_heads, cache_manager.get_head_size(), 4, 32, 4, kv_max_split_num=2)
    block_tables = bb2.add_seqs([seq3])
    cache_manager.allocate_context_from_block_tables(block_tables, bb2.seq_lengths[: bb2.current_batch_size])
    print("bb2> After allocate_context_from_block_tables")
    print(bb2.seq_lengths)
    print(bb2.block_tables)

    unmerged_ids = bb.merge(bb2)
    print("bb> After merge bb2")
    print(bb._sequences_indexes)
    print(bb.seq_lengths)
    print(bb.block_tables)
    print("unmerged_ids: ", unmerged_ids)

    bb.clear(cache_manager.free_block_tables)

    print("bb> After clear")
    print(bb._sequences_indexes)
    print(bb.seq_lengths)
    print(bb.block_tables)

    print(cache_manager._cache_blocks)


if __name__ == "__main__":
    # test_kvcache_manager()

    test_bucket()
