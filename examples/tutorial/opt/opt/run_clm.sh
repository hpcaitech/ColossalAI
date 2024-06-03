set -x
export BS=${1:-16}
export MEMCAP=${2:-0}
export MODEL=${3:-"125m"}
export GPUNUM=${4:-1}

# make directory for logs
mkdir -p ./logs

export MODLE_PATH="facebook/opt-${MODEL}"

# HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
torchrun \
  --nproc_per_node ${GPUNUM} \
  --master_port 19198 \
  run_clm.py \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --output_dir $PWD \
  --mem_cap ${MEMCAP} \
  --model_name_or_path ${MODLE_PATH} \
  --per_device_train_batch_size ${BS} 2>&1 | tee ./logs/colo_${MODEL}_bs_${BS}_cap_${MEMCAP}_gpu_${GPUNUM}.log
