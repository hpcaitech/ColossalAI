export MODEL_NAME= <Your Pretrained Model Path> 
export INSTANCE_DIR= <Your Input Pics Path>
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-save-model"

HF_DATASETS_OFFLINE=1 
TRANSFORMERS_OFFLINE=1 
DIFFUSERS_OFFLINE=1

torchrun --nproc_per_node 2 --master_port=25641 train_dreambooth_colossalai.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of a dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --placement="cuda" \
