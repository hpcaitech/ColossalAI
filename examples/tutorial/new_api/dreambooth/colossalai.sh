HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1 
DIFFUSERS_OFFLINE=1

torchrun --nproc_per_node 4 --master_port=25641 train_dreambooth_colossalai.py \
  --pretrained_model_name_or_path="Path_to_your_model"  \
  --instance_data_dir="Path_to_your_training_image" \
  --output_dir="Path_to_your_save_dir" \
  --instance_prompt="your prompt" \
  --resolution=512 \
  --plugin="gemini" \
  --train_batch_size=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --placement="cuda" \
