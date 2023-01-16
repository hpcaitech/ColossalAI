export DATA=/data/scratch/gpt_data/small-gpt-dataset.json
colossalai run --nproc_per_node=4 train_gpt.py --config ./configs/gpt2_small_zero3_pp1d.py --from_torch
