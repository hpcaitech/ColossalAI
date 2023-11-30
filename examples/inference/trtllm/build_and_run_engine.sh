mode=$1

if [ $mode == "build" ];then
    python build_llama_engine.py --model_dir /llama \
                            --dtype float16 \
                            --use_gpt_attention_plugin float16 \
                            --use_inflight_batching \
                            --paged_kv_cache \
                            --remove_input_padding \
                            --use_gemm_plugin float16 \
                            --output_dir /tensorrtllm_backend/colossal_llama_A800/trt_engines/fp16/2-gpu/ \
                            --world_size 2 \
                            --parallel_build \
                            --tp_size 2
elif [ $mode == "run" ];then
    mpirun -n 2 --allow-run-as-root \
        python run_llama_engine.py  --max_output_len 200 \
                            --tokenizer_dir /llama \
                            --model_name "llama" \
                            --engine_dir /tensorrtllm_backend/colossal_llama_A800/trt_engines/fp16/2-gpu/ \
                            --input_text "Introduce Beijing."
else
    echo "Please set correct mode."
fi
