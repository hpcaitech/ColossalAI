# export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# export INSTANCE_DIR="dog"
# export OUTPUT_DIR="path-to-save-model"

# accelerate launch train_dreambooth.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --instance_prompt="a photo of sks dog" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=1 \
#   --learning_rate=5e-6 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=400 \

python train_dreambooth.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
    --instance_data_dir="/home/lclcq/ColossalAI/applications/stable-diffusion/dreambooth/dog" \
    --output_dir="./weight_output" \
    --instance_prompt="a photo of a dog" \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --learning_rate=5e-6 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --num_class_images=200 \
