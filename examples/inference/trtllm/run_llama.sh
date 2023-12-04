mode=$1
parallel_size=$2

if [[ ${parallel_size} -eq 1 ]];
then
    if [ ${mode} == "build" ];then
        python build_llama_engine.py --model_dir /llama \
                                --dtype float16 \
                                --use_gpt_attention_plugin float16 \
                                --use_inflight_batching \
                                --paged_kv_cache \
                                --remove_input_padding \
                                --use_gemm_plugin float16 \
                                --save_engine \
                                --output_dir /tensorrtllm_backend/colossal_llama_A800/trt_engines/fp16/2-gpu/
    elif [ ${mode} == "run" ];then
        python run_llama_engine.py  --max_output_len 200 \
                            --tokenizer_dir /llama \
                            --model_name "llama" \
                            --engine_dir /tensorrtllm_backend/colossal_llama_A800/trt_engines/fp16/2-gpu/ \
                            --input_text "Introduce Beijing."
    elif [ ${mode} == "build_and_run" ];then
        python build_and_run_llama.py --model_dir /llama \
                           --dtype float16 \
                           --use_gpt_attention_plugin float16 \
                           --use_inflight_batching \
                           --paged_kv_cache \
                           --remove_input_padding \
                           --use_gemm_plugin float16 \
                           --max_output_len 200 \
                           --tokenizer_dir /llama \
                           --model_name "llama" \
                           --input_text "Introduce Beijing."
    else
        echo "Please set correct mode(build/run/build_and_run)."
    fi
elif [ 1 -lt ${parallel_size} ] && [ ${parallel_size} -le 8 ];
then
    if [ ${mode} == "build" ];then
        mpirun -n ${parallel_size} --allow-run-as-root \
            python build_llama_engine.py --model_dir /llama \
                                    --dtype float16 \
                                    --use_gpt_attention_plugin float16 \
                                    --use_inflight_batching \
                                    --paged_kv_cache \
                                    --remove_input_padding \
                                    --use_gemm_plugin float16 \
                                    --save_engine \
                                    --output_dir /tensorrtllm_backend/colossal_llama_A800/trt_engines/fp16/2-gpu/ \
                                    --world_size ${parallel_size} \
                                    --parallel_build \
                                    --tp_size ${parallel_size}
    elif [ ${mode} == "run" ];then
        mpirun -n ${parallel_size} --allow-run-as-root \
            python run_llama_engine.py  --max_output_len 200 \
                                --tokenizer_dir /llama \
                                --model_name "llama" \
                                --engine_dir /tensorrtllm_backend/colossal_llama_A800/trt_engines/fp16/2-gpu/ \
                                --input_text "Introduce Beijing."
    elif [ ${mode} == "build_and_run" ];then
        mpirun -n ${parallel_size} --allow-run-as-root \
            python build_and_run_llama.py --model_dir /llama \
                                    --dtype float16 \
                                    --use_gpt_attention_plugin float16 \
                                    --use_inflight_batching \
                                    --paged_kv_cache \
                                    --remove_input_padding \
                                    --use_gemm_plugin float16 \
                                    --world_size ${parallel_size} \
                                    --tp_size ${parallel_size} \
                                    --max_output_len 200 \
                                    --tokenizer_dir /llama \
                                    --model_name "llama" \
                                    --parallel_build \
                                    --input_text "Introduce Beijing."
    else
        echo "Please set correct mode(build/run/build_and_run)."
    fi
else
    echo "Please set correct parallel_size(1<=parallel_size<=8)."
fi
