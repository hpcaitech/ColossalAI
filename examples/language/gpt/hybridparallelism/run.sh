# load via internet
torchrun --standalone --nproc_per_node 4 --master_port 29800 finetune.py --target_f1 0.6 --plugin hybrid_parallel --model_type "gpt2"

# load from local
# torchrun --standalone --nproc_per_node 4 --master_port 29800 finetune.py --target_f1 0.6 --plugin hybrid_parallel --model_type "gpt2" --pretrained_path "your/path/to/pretrained_model"
