set -xe
pip install -r requirements.txt

export BS=8
export MEMCAP=0
export GPUNUM=1

for BS in 8 32
do
for PLUGIN in "torch_ddp" "torch_ddp_fp16" "low_level_zero" "gemini" "hybrid_parallel"
do

MODEL_PATH="google/vit-base-patch16-224"
torchrun \
  --standalone \
  --nproc_per_node 4 \
  vit_benchmark.py \
  --model_name_or_path ${MODEL_PATH} \
  --mem_cap ${MEMCAP} \
  --plugin ${PLUGIN} \
  --batch_size ${BS}

done
done
