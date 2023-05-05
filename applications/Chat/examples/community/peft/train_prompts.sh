PROMPT_PATH=toys_prompts.txt
PRETRAIN_PATH=toys_pretrain.txt
PRETRAINED_MODEL='/home/yueyulin/pretrained_models/chatglm-6b'
SFT_LORA='toys_sft_lora'
RM_LORA='chatglmrm'
SAVE_PATH='lora_ppo'
#change to THUDM/chatglm-6b if you don't have the pretrained model
torchrun --standalone --nproc_per_nod=1 \
        train_peft_prompts.py \
        --prompt_path $PROMPT_PATH \
        --pretrain_dataset $PRETRAIN_PATH \
        --model chatglm \
        --pretrain $PRETRAINED_MODEL \
        --sft_lora_path $SFT_LORA \
        --rm_lora_path $RM_LORA \
        --save_path $SAVE_PATH \
        --strategy colossalai_zero2 \
        --num_episodes 1 \
        --max_timesteps 4 \
        --update_timesteps 2 \
        --train_batch_size 1