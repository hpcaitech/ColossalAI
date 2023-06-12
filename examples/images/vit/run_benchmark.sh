set -xe
pip install -r requirements.txt

export BS=8
export MEMCAP=0
export GPUNUM=1

for BS in 8 32 128
do
for PLUGIN in "torch_ddp" "torch_ddp_fp16" "low_level_zero" "gemini"
do
for GPUNUM in 1 4
do

MODEL_PATH="google/vit-base-patch16-224"
torchrun \
  --standalone \
  --nproc_per_node ${GPUNUM} \
  vit_benchmark.py \
  --model_name_or_path ${MODEL_PATH} \
  --mem_cap ${MEMCAP} \
  --plugin ${PLUGIN} \
  --batch_size ${BS}
  
done
done
done
