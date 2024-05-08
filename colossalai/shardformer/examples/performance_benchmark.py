"""
Shardformer Benchmark
"""

import torch
import torch.distributed as dist
import transformers
import triton

import colossalai
from colossalai.shardformer import ShardConfig, ShardFormer


def data_gen(batch_size, seq_length):
    input_ids = torch.randint(0, seq_length, (batch_size, seq_length), dtype=torch.long)
    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
    return dict(input_ids=input_ids, attention_mask=attention_mask)


def data_gen_for_sequence_classification(batch_size, seq_length):
    # LM data gen
    # the `labels` of LM is the token of the output, cause no padding, use `input_ids` as `labels`
    data = data_gen(batch_size, seq_length)
    data["labels"] = torch.ones((batch_size), dtype=torch.long)
    return data


MODEL_CONFIG = transformers.LlamaConfig(
    num_hidden_layers=4,
    hidden_size=128,
    intermediate_size=256,
    num_attention_heads=4,
    max_position_embeddings=128,
    num_labels=16,
    pad_token_id=2,
)
BATCH, N_HEADS, N_CTX, D_HEAD = 4, 8, 4096, 64
model_func = lambda: transformers.LlamaForSequenceClassification(MODEL_CONFIG)

# vary seq length for fixed head and batch=4
configs = [
    triton.testing.Benchmark(
        x_names=["N_CTX"],
        x_vals=[2**i for i in range(8, 13)],
        line_arg="provider",
        line_vals=["org_model", "shard_model"],
        line_names=["org_model", "shard_model"],
        styles=[("red", "-"), ("blue", "-")],
        ylabel="ms",
        plot_name=f"lama_for_sequence_classification-batch-{BATCH}",
        args={"BATCH": BATCH, "dtype": torch.float16, "model_func": model_func},
    )
]


def train(model, data):
    output = model(**data)
    loss = output.logits.mean()
    loss.backward()


@triton.testing.perf_report(configs)
def bench_shardformer(BATCH, N_CTX, provider, model_func, dtype=torch.float32, device="cuda"):
    warmup = 10
    rep = 100
    # prepare data
    data = data_gen_for_sequence_classification(BATCH, N_CTX)
    data = {k: v.cuda() for k, v in data.items()}
    model = model_func().to(device)
    model.train()
    if provider == "org_model":
        fn = lambda: train(model, data)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms
    if provider == "shard_model":
        shard_config = ShardConfig(enable_fused_normalization=True, enable_tensor_parallelism=True)
        shard_former = ShardFormer(shard_config=shard_config)
        sharded_model, _ = shard_former.optimize(model)
        sharded_model = sharded_model.cuda()
        fn = lambda: train(sharded_model, data)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms


# start benchmark, command:
# torchrun --standalone --nproc_per_node=2 performance_benchmark.py
if __name__ == "__main__":
    colossalai.launch_from_torch()
    bench_shardformer.run(save_path=".", print_data=dist.get_rank() == 0)
