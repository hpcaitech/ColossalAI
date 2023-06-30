set_n_least_used_CUDA_VISIBLE_DEVICES 1

python train_reward_model.py --pretrain 'microsoft/deberta-v3-large' \
                             --model 'deberta' \
                             --strategy naive \
                             --loss_fn 'log_exp'\
                             --save_path 'rmstatic.pt' \
                             --test True
