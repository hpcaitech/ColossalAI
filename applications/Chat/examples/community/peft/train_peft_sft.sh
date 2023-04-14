TRAIN_SET='toys_sft/train'
SAVE_PATH='toys_sft_lora'
PRETRAINED_MODEL='/home/yueyulin/pretrained_models/chatglm-6b'
#change to THUDM/chatglm-6b if you don't have the pretrained model
torchrun --standalone --nproc_per_node=1 \
        train_peft_sft.py \
        --dataset $TRAIN_SET \
        --model chatglm \
        --pretrain $PRETRAINED_MODEL \
        --save_path $SAVE_PATH \
        --strategy colossalai_zero2 \
        --batch_size 2 \
        --max_epochs 1 