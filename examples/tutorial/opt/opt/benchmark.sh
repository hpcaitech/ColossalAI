export BS=16
export MEMCAP=0
export MODEL="6.7b"
export GPUNUM=1

for MODEL in "6.7b" "13b" "1.3b"
do
for GPUNUM in 8 1
do
for BS in 16 24 32 8
do
for MEMCAP in 0 40
do
pkill -9 torchrun
pkill -9 python

bash ./run_clm.sh $BS $MEMCAP $MODEL $GPUNUM
done
done
done
done
