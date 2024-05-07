import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import colossalai
from colossalai.cluster import DistCoordinator
from colossalai.inference.config import InferenceConfig
from colossalai.inference.core.engine import InferenceEngine
from colossalai.inference.modeling.policy.nopadding_llama import NoPaddingLlamaModelInferPolicy

# For Llama 3, we'll use the following configuration
MODEL_CLS = AutoModelForCausalLM
POLICY_CLS = NoPaddingLlamaModelInferPolicy


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
    model = MODEL_CLS.from_pretrained(model_path_or_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
    tokenizer.pad_token = tokenizer.eos_token
    coordinator.print_on_master(f"Model Config:\n{model.config}")

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
        do_sample=True,
    )
    coordinator.print_on_master(f"Generating...")
    out = engine.generate(prompts=[args.prompt], generation_config=generation_config)
    coordinator.print_on_master(out[0])


# colossalai run --nproc_per_node 1 llama_generation.py -m MODEL_PATH
if __name__ == "__main__":
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="Path to the model or model name")
    parser.add_argument(
        "-p", "--prompt", type=str, default="Introduce some landmarks in the United Kingdom, such as", help="Prompt"
    )
    parser.add_argument("-b", "--max_batch_size", type=int, default=1, help="Max batch size")
    parser.add_argument("-i", "--max_input_len", type=int, default=128, help="Max input length")
    parser.add_argument("-o", "--max_output_len", type=int, default=128, help="Max output length")
    parser.add_argument("-t", "--tp_size", type=int, default=1, help="Tensor Parallelism size")
    parser.add_argument("-d", "--dtype", type=str, default="fp16", help="Data type", choices=["fp16", "fp32", "bf16"])
    parser.add_argument("--use_cuda_kernel", action="store_true", help="Use CUDA kernel, use Triton by default")
    parser.add_argument("--max_length", type=int, default=32, help="Max length for generation")
    args = parser.parse_args()

    infer(args)
