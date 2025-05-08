from vllm import LLM, SamplingParams
import torch
import argparse

parser = argparse.ArgumentParser(description='VLLM args.')
parser.add_argument("-m", "--model_path", type=str, default="/home/duanjunwen/models/Qwen/Qwen2.5-14B", help="The model path. ")
parser.add_argument("-l", "--max_length", type=int, default=8192, help="Max sequence length")
parser.add_argument("-tp", "--tp_size", type=int, default=8, help="Gpu nums")
parser.add_argument("-pp", "--pp_size", type=int, default=2, help="Gpu nums")
parser.add_argument("-t", "--temperature", type=float, default=0.8, help="Temperature")
parser.add_argument("--top_p", type=float, default=0.95, help="Top p")
parser.add_argument("-i", "--input_texts", type=str, default="Find all prime numbers up to 100.", help="Prompts inputs. ")
args = parser.parse_args()

# Create a sampling params object.
sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_length)

# Create an LLM.
llm = LLM(model=args.model_path, max_model_len=args.max_length, tensor_parallel_size=args.tp_size, pipeline_parallel_size=args.pp_size)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(args.input_texts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text}")