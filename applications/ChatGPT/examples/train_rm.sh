set_n_least_used_CUDA_VISIBLE_DEVICES 1

python train_reward_model.py --pretrain '/home/lczht/data2/bloom-560m' \
                             --model 'bloom' \
                             --strategy naive \
                             --loss_fn 'log_exp'\
                             --save_path 'rmstatic.pt' \
                             --test True
