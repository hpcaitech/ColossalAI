set -xe
pip install -r requirements.txt

BS=8
for PLUGIN in "torch_ddp" "torch_ddp_fp16" "low_level_zero" "gemini" "hybrid_parallel"
do

torchrun \
  --standalone \
  --nproc_per_node 4 \
  vit_benchmark.py \
  --model_name_or_path "google/vit-base-patch16-224" \
  --plugin ${PLUGIN} \
  --batch_size ${BS}

done
