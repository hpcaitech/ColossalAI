# Benchmarks

## Benchmark GPT on dummy prompt data

We provide various GPT models (string in parentheses is the corresponding model name used in this script):

- GPT2-S (s)
- GPT2-M (m)
- GPT2-L (l)
- GPT2-XL (xl)
- GPT2-4B (4b)
- GPT2-6B (6b)
- GPT2-8B (8b)
- GPT2-10B (10b)
- GPT2-12B (12b)
- GPT2-15B (15b)
- GPT2-18B (18b)
- GPT2-20B (20b)
- GPT2-24B (24b)
- GPT2-28B (28b)
- GPT2-32B (32b)
- GPT2-36B (36b)
- GPT2-40B (40b)
- GPT3 (175b)

We also provide various training strategies:

- ddp: torch DDP
- colossalai_gemini: ColossalAI GeminiDDP with `placement_policy="cuda"`, like zero3
- colossalai_gemini_cpu: ColossalAI GeminiDDP with `placement_policy="cpu"`, like zero3-offload
- colossalai_zero2: ColossalAI zero2
- colossalai_zero2_cpu: ColossalAI zero2-offload
- colossalai_zero1: ColossalAI zero1
- colossalai_zero1_cpu: ColossalAI zero1-offload

We only support `torchrun` to launch now. E.g.

```shell
# run GPT2-S on single-node single-GPU with min batch size
torchrun --standalone --nproc_per_node 1 benchmark_gpt_dummy.py --model s --strategy ddp --experience_batch_size 1 --train_batch_size 1
# run GPT2-XL on single-node 4-GPU
torchrun --standalone --nproc_per_node 4 benchmark_gpt_dummy.py --model xl --strategy colossalai_zero2
# run GPT3 on 8-node 8-GPU
torchrun --nnodes 8 --nproc_per_node 8 \
 --rdzv_id=$JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$HOST_NODE_ADDR \
 benchmark_gpt_dummy.py --model 175b --strategy colossalai_gemini
```

> ⚠ Batch sizes in CLI args and outputed throughput/TFLOPS are all values of per GPU.

In this benchmark, we assume the model architectures/sizes of actor and critic are the same for simplicity. But in practice, to reduce training cost, we may use a smaller critic.

We also provide a simple shell script to run a set of benchmarks. But it only supports benchmark on single node. However, it's easy to run on multi-nodes by modifying launch command in this script.

Usage:

```shell
# run for GPUS=(1 2 4 8) x strategy=("ddp" "colossalai_zero2" "colossalai_gemini" "colossalai_zero2_cpu" "colossalai_gemini_cpu") x model=("s" "m" "l" "xl" "2b" "4b" "6b" "8b" "10b") x batch_size=(1 2 4 8 16 32 64 128 256)
./benchmark_gpt_dummy.sh
# run for GPUS=2 x strategy=("ddp" "colossalai_zero2" "colossalai_gemini" "colossalai_zero2_cpu" "colossalai_gemini_cpu") x model=("s" "m" "l" "xl" "2b" "4b" "6b" "8b" "10b") x batch_size=(1 2 4 8 16 32 64 128 256)
./benchmark_gpt_dummy.sh 2
# run for GPUS=2 x strategy=ddp x model=("s" "m" "l" "xl" "2b" "4b" "6b" "8b" "10b") x batch_size=(1 2 4 8 16 32 64 128 256)
./benchmark_gpt_dummy.sh 2 ddp
# run for GPUS=2 x strategy=ddp x model=l x batch_size=(1 2 4 8 16 32 64 128 256)
./benchmark_gpt_dummy.sh 2 ddp l
```

## Benchmark OPT with LoRA on dummy prompt data

We provide various OPT models (string in parentheses is the corresponding model name used in this script):

- OPT-125M (125m)
- OPT-350M (350m)
- OPT-700M (700m)
- OPT-1.3B (1.3b)
- OPT-2.7B (2.7b)
- OPT-3.5B (3.5b)
- OPT-5.5B (5.5b)
- OPT-6.7B (6.7b)
- OPT-10B (10b)
- OPT-13B (13b)

We only support `torchrun` to launch now. E.g.

```shell
# run OPT-125M with no lora (lora_rank=0) on single-node single-GPU with min batch size
torchrun --standalone --nproc_per_node 1 benchmark_opt_lora_dummy.py --model 125m --strategy ddp --experience_batch_size 1 --train_batch_size 1 --lora_rank 0
# run OPT-350M with lora_rank=4 on single-node 4-GPU
torchrun --standalone --nproc_per_node 4 benchmark_opt_lora_dummy.py --model 350m --strategy colossalai_zero2 --lora_rank 4
```

> ⚠ Batch sizes in CLI args and outputed throughput/TFLOPS are all values of per GPU.

In this benchmark, we assume the model architectures/sizes of actor and critic are the same for simplicity. But in practice, to reduce training cost, we may use a smaller critic.
