# Benchmarks

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
# run OPT-125M with no lora (lora_rank=0) on single-node single-GPU with min batch size
torchrun --standalone --nproc_per_node 1 benchmark_opt_lora_dummy.py --model 125m --critic_model 125m --strategy ddp --experience_batch_size 1 --train_batch_size 1 --lora_rank 0
# run Actor (OPT-1.3B) and Critic (OPT-350M) with lora_rank=4 on single-node 4-GPU
torchrun --standalone --nproc_per_node 4 benchmark_opt_lora_dummy.py --model 1.3b --critic_model 350m --strategy colossalai_zero2 --lora_rank 4
```
