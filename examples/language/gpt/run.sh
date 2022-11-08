env OMP_NUM_THREADS=16 torchrun --standalone --nproc_per_node=2 train_gpt_demo.py 2>&1 | tee run.log
