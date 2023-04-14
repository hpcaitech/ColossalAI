TRAIN_SET='toys_rewards_train/train'
EVAL_SET='toys_rewards_test/train'
SAVE_PATH='chatglmrm'
PRETRAINED_MODEL='/home/yueyulin/pretrained_models/chatglm-6b'
#change to THUDM/chatglm-6b if you don't have the pretrained model
torchrun --standalone --nproc_per_node=1 \
       train_reward_model.py --pretrain $PRETRAINED_MODEL \
                             --model 'chatglm' \
                             --strategy colossalai_zero2 \
                             --loss_fn 'log_exp'\
                             --save_path $SAVE_PATH \
                             --train_file $TRAIN_SET \
                             --eval_file $EVAL_SET \
                             --valid_file $EVAL_SET \
                             --batch_size 1 \
                             --lora_rank 32 \
                             --max_len 256 \
                             --max_epochs 1