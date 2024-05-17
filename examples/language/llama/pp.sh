export OMP_NUM_THREADS=8
colossalai run --nproc_per_node 4 --master_port 29501 benchmark.py -p 3d --pp 4 -b 8 -g -x --n_chunks 2 --pp_style interleaved
