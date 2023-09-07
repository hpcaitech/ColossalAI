torchrun --standalone --nproc_per_node 4 --master_port 29800 gpt_hybridparallelism.py --target_f1 0.6 --plugin hybrid_parallel --model_type "gpt2"
# torchrun --standalone --nproc_per_node 4 --master_port 29800 gpt_hybridparallelism.py --target_f1 0.6 --plugin hybrid_parallel --model_type "gpt2" --pretrained_path "you/path/to/pretrained_model"
