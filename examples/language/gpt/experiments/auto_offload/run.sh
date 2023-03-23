export BATCH_SIZE=${BATCH_SIZE:-64}
export MODEL_TYPE=${MODEL_TYPE:-"gpt2_medium"}
export MEMORY_BUDGET=${MEMORY_BUDGET:-16}
export SOLVER_TYPE=${SOLVER_TYPE:-"asyn"}

mkdir -p offload_logs

python train_gpt_offload.py --model_type=${MODEL_TYPE} --memory_budget=${MEMORY_BUDGET} --solver_type=${SOLVER_TYPE} --batch_size=${BATCH_SIZE} 2>&1 | tee ./offload_logs/${MODEL_TYPE}_bs_${BATCH_SIZE}_st_${SOLVER_TYPE}.log
