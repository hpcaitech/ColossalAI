# run under /ColossalAI/applications/ColossalChat
export NCCL_SHM_DISABLE=1
export MAX_JOBS=1
export PRETRAINED_MODEL_PATH=./models
export SFT_DATASET=./sft_data
export PROMPT_DATASET=./prompt_data
export PROMPT_RLVR_DATASET=./prompt_data
export PREFERENCE_DATASET=./preference_data
export KTO_DATASET=./kto_data
mkdir models
mkdir sft_data
mkdir prompt_data
mkdir preference_data
mkdir kto_data
# ./tests/test_data_preparation.sh
# ./tests/test_train.sh
