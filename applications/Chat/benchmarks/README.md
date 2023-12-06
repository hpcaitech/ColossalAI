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

- gemini: ColossalAI GeminiPlugin with `placement_policy="cuda"`, like zero3
- gemini_auto: ColossalAI GeminiPlugin with `placement_policy="cpu"`, like zero3-offload
- zero2: ColossalAI zero2
- zero2_cpu: ColossalAI zero2-offload
- 3d: ColossalAI HybridParallelPlugin with TP, DP support

## How to Run
```bash
cd ../tests
# Prepare data for benchmark
SFT_DATASET=/path/to/sft/data/ \
PROMPT_DATASET=/path/to/prompt/data/ \
PRETRAIN_DATASET=/path/to/ptx/data/ \
PREFERENCE_DATASET=/path/to/preference/data \
./test_data_preparation.sh
# Start benchmark
./benchmark_ppo.sh
```
