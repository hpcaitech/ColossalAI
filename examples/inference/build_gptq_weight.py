import argparse
import logging
import os

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import load_dataset
from transformers import AutoTokenizer

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--pretrained_model_dir",
    type=str,
    help="pretrained model directory",
)
parser.add_argument(
    "--quantized_model_dir",
    help="the path of out gptq model",
    type=str,
)
parser.add_argument(
    "--bits",
    type=int,
    default=4,
    help="quantize model bits",
)
parser.add_argument(
    "--group_size",
    type=int,
    default=128,
    help="the group size of model",
)
parser.add_argument(
    "--dataset-path",
    type=str,
    help="location of the  dataset",
)
args = parser.parse_args()

pretrained_model_dir = args.pretrained_model_dir
quantized_model_dir = args.quantized_model_dir

quantize_config = BaseQuantizeConfig(
    bits=args.bits,  # quantize model to 4-bit
    group_size=args.group_size,  # it is recommended to set the value to 128
)

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir)


if args.dataset_path is not None:
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Cannot find the dataset at {args.dataset_path}")
    else:
        dataset = load_dataset("json", data_files=args.dataset_path, split="train")
else:
    dataset = [
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
    ]

examples = [
    tokenizer(
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
    )
]

# load un-quantized model, by default, the model will always be loaded into CPU memory
model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)
model.config.pad_token_id = model.config.eos_token_id

# quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
model.quantize(examples)

# save quantized model
model.save_quantized(quantized_model_dir)
