HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1
DIFFUSERS_OFFLINE=1

torchrun --nproc_per_node 4 --standalone stable_diffusion_colossalai_trainer.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="/home/lclcq/ColossalAI/applications/stable-diffusion/text_img2img/dog" \
  --output_dir="./weight_output" \
  --instance_prompt="a picture of a dog" \
  --resolution=512 \
  --plugin="gemini" \
  --train_batch_size=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --placement="cuda" \
  --task_type="dreambooth" 
  