HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1
DIFFUSERS_OFFLINE=1

python main.py --logdir /tmp --train --base configs/Teyvat/train_colossalai_teyvat.yaml --ckpt diffuser_root_dir/512-base-ema.ckpt
