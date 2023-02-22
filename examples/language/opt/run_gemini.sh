set -x
export BS=${BS:-16}
export MEMCAP=${MEMCAP:-0}
# Acceptable values include `125m`, `350m`, `1.3b`, `2.7b`, `6.7b`, `13b`, `30b`, `66b`. For `175b`
export MODEL=${MODEL:-"125m"}
export GPUNUM=${GPUNUM:-1}

# make directory for logs
mkdir -p ./logs

export MODLE_PATH="facebook/opt-${MODEL}"

# HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
torchrun \
  --nproc_per_node ${GPUNUM} \
  --master_port 19198 \
  train_gemini_opt.py \
  --mem_cap ${MEMCAP} \
  --model_name_or_path ${MODLE_PATH} \
  --batch_size ${BS} 2>&1 | tee ./logs/colo_${MODEL}_bs_${BS}_cap_${MEMCAP}_gpu_${GPUNUM}.log
