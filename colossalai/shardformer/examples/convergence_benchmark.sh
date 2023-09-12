torchrun --standalone --nproc_per_node=4 convergence_benchmark.py \
    --model "bert" \
    --pretrain "bert-base-uncased" \
    --max_epochs 3 \
    --batch_size 2 \
    --lr 2.4e-5 \
    --fused_layernorm False \
    --accumulation_steps 8 \
    --warmup_fraction 0.03
