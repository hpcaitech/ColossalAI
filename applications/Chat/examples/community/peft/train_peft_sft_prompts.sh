
torchrun --standalone --nproc_per_node=1 \
        train_peft_sft.py \
        --dataset /home/yueyulin/ask_law_data_dir/ask_law_sft_train/train/ \
        --model chatglm \
        --pretrain /home/yueyulin/pretrained_models/chatglm-6b \
        --model_path /home/yueyulin/models/sft_law_chatglm6b_ask_law_prompts \
        --strategy colossalai_zero2 \
        --batch_size 2 \
        --max_epochs 1 \
        --save_path /home/yueyulin/models/sft_law_chatglm6b_ask_law_prompts 