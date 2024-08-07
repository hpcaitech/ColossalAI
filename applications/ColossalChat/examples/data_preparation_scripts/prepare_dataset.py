#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare dataset scripts

Usage:
- For SFT dataset preparation (SFT)
python prepare_dataset.py --type sft \
    --data_input_dirs /PATH/TO/SFT/DATASET \
    --conversation_template_config /PATH/TO/CHAT/TEMPLATE/CONFIG.json \
    --tokenizer_dir  "" \
    --data_cache_dir $SAVE_DIR/cache \
    --data_jsonl_output_dir $SAVE_DIR/jsonl \
    --data_arrow_output_dir $SAVE_DIR/arrow \

- For prompt dataset preparation (PPO)
python prepare_dataset.py --type prompt \
    --data_input_dirs /PATH/TO/SFT/DATASET \
    --conversation_template_config /PATH/TO/CHAT/TEMPLATE/CONFIG.json \
    --tokenizer_dir  "" \
    --data_cache_dir $SAVE_DIR/cache \
    --data_jsonl_output_dir $SAVE_DIR/jsonl \
    --data_arrow_output_dir $SAVE_DIR/arrow \

- For Preference dataset preparation (DPO and Reward model training)
python prepare_dataset.py --type preference \
    --data_input_dirs /PATH/TO/SFT/DATASET \
    --conversation_template_config /PATH/TO/CHAT/TEMPLATE/CONFIG.json \
    --tokenizer_dir  "" \
    --data_cache_dir $SAVE_DIR/cache \
    --data_jsonl_output_dir $SAVE_DIR/jsonl \
    --data_arrow_output_dir $SAVE_DIR/arrow \
"""

import argparse
import json
import math
import os
import random
import time
from multiprocessing import cpu_count

from coati.dataset import setup_conversation_template, tokenize_kto, tokenize_prompt, tokenize_rlhf, tokenize_sft
from datasets import dataset_dict, load_dataset
from transformers import AutoTokenizer

from colossalai.logging import get_dist_logger

logger = get_dist_logger()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        default=None,
        choices=["sft", "prompt", "preference", "kto"],
        help="Type of dataset, chose from 'sft', 'prompt', 'preference'. 'kto'",
    )
    parser.add_argument(
        "--data_input_dirs",
        type=str,
        required=True,
        default=None,
        help="Comma(i.e., ',') separated list of all data directories containing `.jsonl` data files.",
    )
    parser.add_argument(
        "--tokenizer_dir", type=str, required=True, default=None, help="A directory containing the tokenizer"
    )
    parser.add_argument(
        "--conversation_template_config",
        type=str,
        default="conversation_template_config",
        help="Path \
        to save conversation template config files.",
    )
    parser.add_argument("--data_cache_dir", type=str, default="cache", help="Data cache directory")
    parser.add_argument(
        "--data_jsonl_output_dir",
        type=str,
        default="jsonl_output",
        help="Output directory of spliced dataset with jsonl format",
    )
    parser.add_argument(
        "--data_arrow_output_dir",
        type=str,
        default="arrow_output",
        help="Output directory of spliced dataset with arrow format",
    )
    parser.add_argument("--max_length", type=int, default=4096, help="Max length of each spliced tokenized sequence")
    parser.add_argument("--num_spliced_dataset_bins", type=int, default=10, help="Number of spliced dataset bins")
    parser.add_argument(
        "--num_samples_per_datafile",
        type=int,
        default=-1,
        help="Number of samples to be generated from each data file. -1 denote all samples.",
    )
    args = parser.parse_args()

    if args.num_spliced_dataset_bins >= 100000:
        raise ValueError("Too many spliced divisions, must be smaller than 100000")

    assert not os.path.exists(args.data_cache_dir), f"Find existed data cache dir {args.data_cache_dir}"
    assert not os.path.exists(
        args.data_jsonl_output_dir
    ), f"Find existed jsonl data output dir {args.data_jsonl_output_dir}"
    assert not os.path.exists(
        args.data_arrow_output_dir
    ), f"Find existed arrow data output dir {args.data_arrow_output_dir}"
    os.makedirs(args.data_jsonl_output_dir)
    os.makedirs(args.data_arrow_output_dir)

    # Prepare to all input datasets
    input_data_paths = []
    input_data_dirs = args.data_input_dirs.split(",")
    for ds_dir in input_data_dirs:
        ds_dir = os.path.abspath(ds_dir)
        assert os.path.exists(ds_dir), f"Not find data dir {ds_dir}"
        ds_files = [name for name in os.listdir(ds_dir) if name.endswith(".jsonl")]
        ds_paths = [os.path.join(ds_dir, name) for name in ds_files]
        input_data_paths.extend(ds_paths)

    # Prepare to data splitting.
    train_splits = []
    split_interval = math.ceil(100 / args.num_spliced_dataset_bins)
    for i in range(0, 100, split_interval):
        start = i
        end = i + split_interval
        if end > 100:
            end = 100
        train_splits.append(f"train[{start}%:{end}%]")

    # Prepare the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=False, trust_remote_code=True)
    if os.path.exists(args.conversation_template_config):
        chat_template_config = json.load(open(args.conversation_template_config, "r", encoding="utf8"))
    else:
        chat_template_config = {
            "system_message": "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        }  # Use default system message
    if args.type == "preference":
        if "stop_ids" not in chat_template_config:
            # Ask the user to define stop_ids for PPO training
            dummy_messages = [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
                {"role": "user", "content": "Who made you?"},
                {"role": "assistant", "content": "I am a chatbot trained by Colossal-AI."},
            ]
            dummy_prompt = tokenizer.apply_chat_template(dummy_messages, tokenize=False)
            tokenized = tokenizer(dummy_prompt, add_special_tokens=False)["input_ids"]
            tokens = tokenizer.convert_ids_to_tokens(tokenized, skip_special_tokens=False)
            corresponding_str = [tokenizer.convert_tokens_to_string([token]) for token in tokens]
            token_id_mapping = [{"token": s, "id": tokenized[i]} for i, s in enumerate(corresponding_str)]
            stop_ids = input(
                "For PPO, we recommend to provide stop_ids for the properly stop the generation during roll out stage. "
                "stop_ids are the ids of repetitive pattern that indicate the end of the assistant's response. "
                "Here is an example of formatted prompt and token-id mapping, you can set stop_ids by entering a list "
                "of integers, separate by space, press `Enter` to end. Or you can press `Enter` without input if you are "
                "not using PPO or you prefer to not set the stop_ids, in that case, stop_ids will be set to tokenizer.eos_token_id. "
                f"\nPrompt:\n{dummy_prompt}\nToken-id Mapping:\n{token_id_mapping}\nstop_ids:"
            )
            if stop_ids == "":
                chat_template_config["stop_ids"] = [tokenizer.eos_token_id]
            else:
                try:
                    chat_template_config["stop_ids"] = [int(s) for s in stop_ids.split()]
                except ValueError:
                    raise ValueError("Invalid input, please provide a list of integers.")
    else:
        # Set stop_ids to eos_token_id for other dataset types if not exist
        if "stop_ids" not in chat_template_config:
            chat_template_config["stop_ids"] = [tokenizer.eos_token_id]

    conversation_template = setup_conversation_template(
        tokenizer, chat_template_config=chat_template_config, save_path=args.conversation_template_config
    )
    if hasattr(tokenizer, "pad_token") and hasattr(tokenizer, "eos_token") and tokenizer.eos_token is not None:
        try:
            # Some tokenizers doesn't allow to set pad_token mannually e.g., Qwen
            tokenizer.pad_token = tokenizer.eos_token
        except AttributeError as e:
            logger.warning(f"Unable to set pad token to eos token, {str(e)}")
    if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
        logger.warning(
            "The tokenizer does not have a pad token which is required. May lead to unintended behavior in training, Please consider manually set them."
        )

    list_dataset = load_dataset(
        path="json",
        data_files=input_data_paths,
        cache_dir=os.path.join(args.data_cache_dir, "raw"),
        keep_in_memory=False,
        split=train_splits,
        num_proc=cpu_count(),
    )

    if args.type == "sft":
        preparation_function = tokenize_sft
    elif args.type == "prompt":
        preparation_function = tokenize_prompt
    elif args.type == "preference":
        preparation_function = tokenize_rlhf
    elif args.type == "kto":
        preparation_function = tokenize_kto
    else:
        raise ValueError("Unknow dataset type. Please choose one from ['sft', 'prompt', 'preference']")

    for index, dataset in enumerate(list_dataset):
        assert isinstance(dataset, dataset_dict.Dataset)
        if len(dataset) == 0:
            # Hack: Skip empty dataset. If dataset contains less than num_of_rank samples, some rank may have empty dataset and leads to error
            continue
        if args.num_samples_per_datafile > 0:
            # limit the number of samples in each dataset
            dataset = dataset.select(
                random.sample(range(len(dataset)), min(args.num_samples_per_datafile, len(dataset)))
            )
        logger.info(f"Start to process part-{index}/{len(list_dataset)} of all original datasets.")
        dataset = dataset.map(
            function=preparation_function,
            fn_kwargs={
                "tokenizer": tokenizer,
                "conversation_template": conversation_template,
                "max_length": args.max_length,
            },
            keep_in_memory=False,
            num_proc=min(len(dataset), cpu_count()),
        )
        if args.type == "kto":
            filter_by = "completion"
        elif args.type == "preference":
            filter_by = "chosen_input_ids"
        else:
            filter_by = "input_ids"
        dataset = dataset.filter(lambda data: data[filter_by] is not None)

        # Save each jsonl spliced dataset.
        output_index = "0" * (5 - len(str(index))) + str(index)
        output_name = f"part-{output_index}"
        output_jsonl_path = os.path.join(args.data_jsonl_output_dir, output_name + ".jsonl")
        st = time.time()
        with open(file=output_jsonl_path, mode="w", encoding="utf-8") as fp_writer:
            count = 0
            for data_point in dataset:
                if count % 500 == 0:
                    logger.info(f"processing {count} spliced data points for {fp_writer.name}")
                count += 1
                fp_writer.write(json.dumps(data_point, ensure_ascii=False) + "\n")
        logger.info(
            f"Current file {fp_writer.name}; "
            f"Data size: {len(dataset)}; "
            f"Time cost: {round((time.time() - st) / 60, 6)} minutes."
        )
        # Save each arrow spliced dataset
        output_arrow_path = os.path.join(args.data_arrow_output_dir, output_name)
        logger.info(f"Start to save {output_arrow_path}")
        dataset = load_dataset(
            path="json",
            data_files=[output_jsonl_path],
            cache_dir=os.path.join(args.data_cache_dir, "tokenized"),
            keep_in_memory=False,
            num_proc=cpu_count(),
            split="train",
        )
        dataset.save_to_disk(dataset_path=output_arrow_path, num_proc=min(len(dataset), cpu_count()))


if __name__ == "__main__":
    main()
