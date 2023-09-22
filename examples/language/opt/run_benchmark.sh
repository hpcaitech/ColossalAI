set -xe
pip install -r requirements.txt

export BS=32
export MEMCAP=0
export GPUNUM=1

# acceptable values include `125m`, `350m`, `1.3b`, `2.7b`, `6.7b`, `13b`, `30b`, `66b`
export MODEL="125m"

for BS in 8 32 128
do
for PLUGIN in "torch_ddp" "torch_ddp_fp16" "low_level_zero" "gemini"
do
for GPUNUM in 1 4
do

MODLE_PATH="facebook/opt-${MODEL}"
colossalai run \
  --nproc_per_node ${GPUNUM} \
  --master_port 29505 \
  opt_benchmark.py \
  --model_name_or_path ${MODLE_PATH} \
  --mem_cap ${MEMCAP} \
  --plugin ${PLUGIN} \
  --batch_size ${BS}

done
done
done
