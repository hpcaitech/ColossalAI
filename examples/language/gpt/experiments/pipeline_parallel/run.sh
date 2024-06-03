export GPUNUM=${GPUNUM:-4}
export BATCH_SIZE=${BATCH_SIZE:-16}
export MODEL_TYPE=${MODEL_TYPE:-"gpt2_medium"}
export NUM_MICROBATCH=${NUM_MICROBATCH:-8}

mkdir -p pp_logs
python train_gpt_pp.py --device="cuda" --model_type=${MODEL_TYPE} --num_microbatches=${NUM_MICROBATCH} --world_size=${GPUNUM} --batch_size=${BATCH_SIZE} 2>&1 | tee ./pp_logs/${MODEL_TYPE}_gpu_${GPUNUM}_bs_${BATCH_SIZE}_nm_${NUM_MICROBATCH}.log
