HF_DATASETS_OFFLINE=1 
TRANSFORMERS_OFFLINE=1 

python main.py --logdir /tmp -t --postfix test -b configs/train_colossalai.yaml 
