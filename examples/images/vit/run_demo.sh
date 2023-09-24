set -xe
pip install -r requirements.txt

# model name or path
MODEL="google/vit-base-patch16-224"

# path for saving model
OUTPUT_PATH="./output_model"

# plugin(training strategy)
# can only be one of "torch_ddp"/"torch_ddp_fp16"/"low_level_zero"/"gemini"/"hybrid_parallel"
PLUGIN="gemini"
#PLUGIN="hybrid_parallel"

# configuration of parallel group sizes, only used when setting PLUGIN to "hybrid_parallel"
TP_SIZE=2
PP_SIZE=2

# number of gpus to use
GPUNUM=4

# batch size per data parallel group
BS=16

# learning rate
LR="2e-4"

# number of epoch
EPOCH=3

# weight decay
WEIGHT_DECAY=0.05

# ratio of warmup steps
WARMUP_RATIO=0.3

# run the script for demo
colossalai run \
  --nproc_per_node ${GPUNUM} \
  --master_port 29505 \
  vit_train_demo.py \
  --model_name_or_path ${MODEL} \
  --output_path ${OUTPUT_PATH} \
  --plugin ${PLUGIN} \
  --batch_size ${BS} \
  --tp_size ${TP_SIZE} \
  --pp_size ${PP_SIZE} \
  --num_epoch ${EPOCH} \
  --learning_rate ${LR} \
  --weight_decay ${WEIGHT_DECAY} \
  --warmup_ratio ${WARMUP_RATIO}
