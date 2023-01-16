$(cd `dirname $0`;pwd)

for BATCH_SIZE in 2
do
for GPUNUM in 1 4
do
env OMP_NUM_THREADS=12 torchrun  --standalone --nproc_per_node=${GPUNUM}  --master_port 29501  train.py --dummy_data=True --batch_size=${BATCH_SIZE}  2>&1 | tee run.log
done
done
