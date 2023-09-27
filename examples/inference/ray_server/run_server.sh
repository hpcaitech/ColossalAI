export CUDA_VISIBLE_DEVICES=0,1,2,3
python api_server_async_engine.py \
       --model decapoda-research/llama-7b-hf \
       --tokenizer hf-internal-testing/llama-tokenizer \
       -tp 4 \
       --host localhost \
       --port 8000
