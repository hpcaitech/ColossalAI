env OMP_NUM_THREADS=16 torchrun --standalone --nproc_per_node=4 train_gpt_demo.py --tp_degree=2 --placement='cpu' 2>&1 | tee run.log
