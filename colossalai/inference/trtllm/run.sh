python test_llama_build.py --model_dir /llama \
                           --dtype float16 \
                           --use_gpt_attention_plugin float16 \
                           --use_inflight_batching \
                           --paged_kv_cache \
                           --remove_input_padding \
                           --use_gemm_plugin float16 \
                           --output_dir /tensorrtllm_backend/colossal_llama_A800/trt_engines/fp16/2-gpu/ \
                           --world_size 2 \
                           --tp_size 2
