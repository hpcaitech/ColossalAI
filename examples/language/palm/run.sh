# distplan in ["colossalai", "pytorch"]
export DISTPAN="colossalai"

# The following options only valid when DISTPAN="colossalai"
export TPDEGREE=1
export GPUNUM=4
export PLACEMENT='cpu'
export USE_SHARD_INIT=False
export BATCH_SIZE=1

env OMP_NUM_THREADS=12 colossalai run --nproc_per_node ${GPUNUM} --master_port 29505  train.py  \
--dummy_data=True --tp_degree=${TPDEGREE} --batch_size=${BATCH_SIZE} --plugin='gemini' \
--placement ${PLACEMENT} --shardinit ${USE_SHARD_INIT} --distplan ${DISTPAN} 2>&1 | tee run.log
