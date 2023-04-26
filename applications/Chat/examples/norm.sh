# set_n_least_used_CUDA_VISIBLE_DEVICES 4

# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port 29500 --nproc_per_node=4 \
#              train_reward_model.py --pretrain '/data/scratch/alpaca-7B' \
#                              --model 'bloom' \
#                              --strategy colossalai_zero2 \
#                              --loss_fn 'log_exp'\
#                              --save_path '/home/lczht/data2/Coati/examples/rm_bloomz_1b7.pt' \
#                              --dataset 'Anthropic/hh-rlhf'\
#                              --subset 'harmless-base'\
#                              --test True

CUDA_VISIBLE_DEVICES=5 python normalize_rm.py --pretrain  /home/lczht/data2/bloom-560m \
                             --model 'bloom' \
                             --model_path '/home/lczht/data2/Coati/examples/rm_bloom560m.pt' \
                             --dataset 'Anthropic/hh-rlhf'\
                             --test True \