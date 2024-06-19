import argparse

from torch import bfloat16, float16, float32
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import colossalai
from colossalai.cluster import DistCoordinator
from colossalai.inference.config import InferenceConfig
from colossalai.inference.core.engine import InferenceEngine
from colossalai.inference.modeling.policy.nopadding_llama import NoPaddingLlamaModelInferPolicy

# For Llama 3, we'll use the following configuration
MODEL_CLS = AutoModelForCausalLM
POLICY_CLS = NoPaddingLlamaModelInferPolicy

TORCH_DTYPE_MAP = {
    "fp16": float16,
    "fp32": float32,
    "bf16": bfloat16,
}


def infer(args):
    # ==============================
    # Launch colossalai, setup distributed environment
    # ==============================
    colossalai.launch_from_torch()
    coordinator = DistCoordinator()

    # ==============================
    # Load model and tokenizer
    # ==============================
    model_path_or_name = args.model
    model = MODEL_CLS.from_pretrained(model_path_or_name, torch_dtype=TORCH_DTYPE_MAP.get(args.dtype, None))
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
    tokenizer.pad_token = tokenizer.eos_token
    # coordinator.print_on_master(f"Model Config:\n{model.config}")

    # ==============================
    # Initialize InferenceEngine
    # ==============================
    inference_config = InferenceConfig(
        dtype=args.dtype,
        max_batch_size=args.max_batch_size,
        max_input_len=args.max_input_len,
        max_output_len=args.max_output_len,
        prefill_ratio=1.2,
        block_size=16,
        tp_size=args.tp_size,
        use_cuda_kernel=args.use_cuda_kernel,
        enable_streamingllm=args.enable_streamingllm,
        start_token_size=args.start_token_size,
        generated_token_size=args.generated_token_size,
    )
    coordinator.print_on_master(f"Initializing Inference Engine...")
    engine = InferenceEngine(model, tokenizer, inference_config, model_policy=POLICY_CLS(), verbose=True)

    # ==============================
    # Generation
    # ==============================
    generation_config = GenerationConfig(
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_length=args.max_length,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        repetition_penalty=args.repetition_penalty,
    )
    coordinator.print_on_master(f"Generating...")
    out = engine.generate(prompts=[args.prompt], generation_config=generation_config)
    coordinator.print_on_master(out)

    # ==============================
    # Optionally, load drafter model and proceed speculative decoding
    # ==============================
    drafter_model_path_or_name = args.drafter_model
    if drafter_model_path_or_name is not None:
        drafter_model = AutoModelForCausalLM.from_pretrained(drafter_model_path_or_name)
        # turn on speculative decoding with the drafter model
        engine.enable_spec_dec(drafter_model)
        coordinator.print_on_master(f"Generating...")
        out = engine.generate(prompts=[args.prompt], generation_config=generation_config)
        coordinator.print_on_master(out)

        engine.disable_spec_dec()


# colossalai run --nproc_per_node 1 llama_generation.py -m MODEL_PATH
# colossalai run --nproc_per_node 2 llama_generation.py -m MODEL_PATH --tp_size 2
if __name__ == "__main__":
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="Path to the model or model name")
    parser.add_argument("--drafter_model", type=str, help="Path to the drafter model or model name")
    parser.add_argument(
        "-p", "--prompt", type=str, default="Introduce some landmarks in the United Kingdom, such as", help="Prompt"
    )
    parser.add_argument("-b", "--max_batch_size", type=int, default=1, help="Max batch size")
    parser.add_argument("-i", "--max_input_len", type=int, default=128, help="Max input length")
    parser.add_argument("-o", "--max_output_len", type=int, default=128, help="Max output length")
    parser.add_argument("-t", "--tp_size", type=int, default=1, help="Tensor Parallelism size")
    parser.add_argument("-d", "--dtype", type=str, default="fp16", help="Data type", choices=["fp16", "fp32", "bf16"])
    parser.add_argument("--use_cuda_kernel", action="store_true", help="Use CUDA kernel, use Triton by default")
    # Generation configs
    parser.add_argument("--max_length", type=int, default=64, help="Max length for generation")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling for generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for generation")
    parser.add_argument("--top_k", type=int, default=50, help="Top k for generation")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top p for generation")
    parser.add_argument("--enable_streamingllm", action="store_true", help="Whether to use StreamingLLM")
    parser.add_argument(
        "--start_token_size", type=int, default=4, help="The size of the start_token, When using StreamingLLM,"
    )
    parser.add_argument(
        "--generated_token_size", type=int, default=512, help="The size of the generated_token, When using StreamingLLM"
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=0,
        help="If no_repeat_ngram_size > 0, the consecutive tokens of ngram size can only appear once in inference sentences.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="The parameter that influences the model's treatment of new tokens in relation to their appearance in the prompt and the generated text. Values greater than 1 incentivize the model to introduce new tokens, whereas values less than 1 incentivize token repetition., defaults to 1.0.",
    )
    args = parser.parse_args()

    infer(args)
