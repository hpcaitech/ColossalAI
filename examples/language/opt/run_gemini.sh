set -x
export BS=${BS:-16}
export MEMCAP=${MEMCAP:-0}
export MODEL=${MODEL:-"125m"}
export GPUNUM=${GPUNUM:-1}

# make directory for logs
mkdir -p ./logs

export MODLE_PATH="facebook/opt-${MODEL}"

# HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
torchrun \
  --nproc_per_node ${GPUNUM} \
  --master_port 19198 \
  run_clm.py \
  --mem_cap ${MEMCAP} \
  --model_name_or_path ${MODLE_PATH} \
  --batch_size ${BS} 2>&1 | tee ./logs/colo_${MODEL}_bs_${BS}_cap_${MEMCAP}_gpu_${GPUNUM}.log
