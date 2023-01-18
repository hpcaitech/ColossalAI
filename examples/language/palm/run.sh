# distplan in ["colossalai", "pytorch"]
export DISTPAN="colossalai"

# The following options only valid when DISTPAN="colossalai"
export TPDEGREE=1
export GPUNUM=1
export PLACEMENT='cpu'
export USE_SHARD_INIT=False
export BATCH_SIZE=4

env OMP_NUM_THREADS=12 torchrun  --standalone --nproc_per_node=${GPUNUM}  --master_port 29501  train.py  --tp_degree=${TPDEGREE} --batch_size=${BATCH_SIZE} --placement ${PLACEMENT} --shardinit ${USE_SHARD_INIT} --distplan ${DISTPAN} 2>&1 | tee run.log
