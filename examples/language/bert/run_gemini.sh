set -x
# distplan in ["CAI_ZeRO1", "CAI_ZeRO2", "CAI_Gemini", "Pytorch_DDP", "Pytorch_ZeRO"]
export DISTPLAN=${DISTPLAN:-"CAI_Gemini"}

# The following options only valid when DISTPLAN="colossalai"
export GPUNUM=${GPUNUM:-1}
export PLACEMENT=${PLACEMENT:-"cpu"}
export BATCH_SIZE=${BATCH_SIZE:-16}

# bert | albert
export MODEL_TYPE=${MODEL_TYPE:-"bert"}
export TRAIN_STEP=${TRAIN_STEP:-10}

mkdir -p gemini_logs

env CUDA_LAUNCH_BLOCKING=1 torchrun --standalone --nproc_per_node=${GPUNUM} ./train_bert_demo.py \
--model_type=${MODEL_TYPE} \
--batch_size=${BATCH_SIZE} \
--placement=${PLACEMENT} \
--distplan=${DISTPLAN} \
--train_step=${TRAIN_STEP} \
2>&1 | tee ./gemini_logs/${MODEL_TYPE}_${DISTPLAN}_gpu_${GPUNUM}_bs_${BATCH_SIZE}_${PLACEMENT}.log
