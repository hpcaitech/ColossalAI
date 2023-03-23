set -x
# distplan in ["CAI_ZeRO1", "CAI_ZeRO2", "CAI_Gemini", "Pytorch_DDP", "Pytorch_ZeRO"]
export DISTPLAN=${DISTPLAN:-"CAI_Gemini"}

# The following options only valid when DISTPLAN="colossalai"
export GPUNUM=${GPUNUM:-1}
export TPDEGREE=${TPDEGREE:-1}
export PLACEMENT=${PLACEMENT:-"cpu"}
export USE_SHARD_INIT=${USE_SHARD_INIT:-False}
export BATCH_SIZE=${BATCH_SIZE:-16}
export MODEL_TYPE=${MODEL_TYPE:-"gpt2_medium"}
export TRAIN_STEP=${TRAIN_STEP:-10}
# export PYTHONPATH=$PWD:$PYTHONPATH

if [ ${USE_SHARD_INIT} = "True" ]; then
  USE_SHARD_INIT="--shardinit"
else
  USE_SHARD_INIT=""
fi

mkdir -p gemini_logs

torchrun --standalone --nproc_per_node=${GPUNUM} ./train_gpt_demo.py \
--tp_degree=${TPDEGREE} \
--model_type=${MODEL_TYPE} \
--batch_size=${BATCH_SIZE} \
--placement=${PLACEMENT} \
${USE_SHARD_INIT} \
--distplan=${DISTPLAN} \
--train_step=${TRAIN_STEP} \
2>&1 | tee ./gemini_logs/${MODEL_TYPE}_${DISTPLAN}_gpu_${GPUNUM}_bs_${BATCH_SIZE}_tp_${TPDEGREE}_${PLACEMENT}.log
