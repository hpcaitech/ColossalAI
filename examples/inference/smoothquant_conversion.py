import argparse
import os

import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer

from colossalai.inference.quant.smoothquant.models.llama import convert_llama_to_smoothquant


def build_model_and_tokenizer(model_name):
    tokenizer = LlamaTokenizer.from_pretrained(model_name, model_max_length=512)
    kwargs = {"torch_dtype": torch.float16, "device_map": "sequential"}
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model = model.to(torch.float32)
    # config = {
    #     "architectures": ["LLaMAForCausalLM"],
    #     "bos_token_id": 0,
    #     "eos_token_id": 1,
    #     "hidden_act": "silu",
    #     "hidden_size": 4096,
    #     "initializer_range": 0.02,
    #     "intermediate_size": 11008,
    #     "max_position_embeddings": 2048,
    #     "max_sequence_length": 2048,
    #     "model_type": "llama",
    #     "num_attention_heads": 32,
    #     "num_hidden_layers": 2,
    #     "num_key_value_heads": 32,
    #     "pad_token_id": -1,
    #     "pretraining_tp": 1,
    #     "rms_norm_eps": 1e-06,
    #     "torch_dtype": "float32",
    #     "transformers_version": "4.32.1",
    #     "use_cache": True,
    #     "vocab_size": 32000,
    # }
    # config = LlamaConfig(**config)
    # model = LlamaForCausalLM(config)

    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-1.3b", help="model name")
    parser.add_argument(
        "--output-path",
        type=str,
        default="act_scales/opt-1.3b.pt",
        help="where to save the act scales",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="dataset/val.jsonl.zst",
        help="location of the calibration dataset, we use the validation set of the Pile dataset",
    )
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=512)
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = parse_args()
    model_path = "/home/lcxk/data3/llama-7b-hf"
    dataset_path = "/home/lcxk/data3/datasets/cc_news.json"
    num_samples = 10
    seq_len = 512
    print("data path", dataset_path)
    data_files = {"train": dataset_path}

    # # dataset = load_dataset("/home/lcxk/data3/datasets/cc_news.py", data_files=dataset_path)
    # dataset = load_dataset("json", data_files=dataset_path)

    # print("text", dataset["train"]["rows"][0][1``]["row"]["text"])
    # # for test in dataset["train"]["rows"]:
    # #     print(test)
    # dataset = dataset.shuffle(seed=42)
    # # print("text", dataset["rows"][0])

    model, tokenizer = build_model_and_tokenizer(model_path)
    print("config:", model.config)
    if not os.path.exists(dataset_path):
        print(f"Cannot find the dataset at {args.dataset_path}")
        print("Please download the Pile dataset and put the validation set at the path")
        print(
            "You can download the validation dataset of the Pile at https://mystic.the-eye.eu/public/AI/pile/val.jsonl.zst"
        )
        raise FileNotFoundError

    # act_scales = get_act_scales(model, tokenizer, dataset_path, num_samples, seq_len)

    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # torch.save(act_scales, output_path)

    # act_scales = torch.load(output_path)
    # smooth_lm(model, act_scales, 0.5)
    # # tokenizer = AutoTokenizer.from_pretrained(model_path)

    if not os.path.exists(dataset_path):
        print(f"Cannot find the dataset at (dataset_path)")
        print("Please download the Pile dataset and put the validation set at the path")
        print(
            "You can download the validation dataset of the Pile at https://mystic.the-eye.eu/public/AI/pile/val.jsonl.zst"
        )
        raise FileNotFoundError

    decoder_layer_scales, raw_scales = convert_llama_to_smoothquant(
        model, tokenizer, dataset_path, num_samples=num_samples, seq_len=seq_len
    )
    model = model.cuda()
    generate_kwargs = dict(max_new_tokens=16, do_sample=False, use_cache=True)

    print("decoder  layer scales:", decoder_layer_scales)
    # output_path = Path(args.output_path) / (Path(args.model_name).name + "-smoothquant.pt")
    input_tokens = tokenizer(["New York City"], return_tensors="pt").to(model.device)

    max_batch_size = 1
    max_input_len = 7
    input_tokens = {
        "input_ids": torch.randint(1, 1000, (max_batch_size, max_input_len), device="cuda"),
        "attention_mask": torch.ones((max_batch_size, max_input_len), device="cuda"),
    }

    print("input:", input)
    # gen_args = {"max_input_"}
    out = model.generate(**input_tokens, **generate_kwargs)
    text = tokenizer.batch_decode(out)
    print("text:", text)


if __name__ == "__main__":
    main()
