#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare sft dataset for finetuning
"""

import argparse
import json
import math
import os
import random
from multiprocessing import cpu_count

from coati.dataset import setup_conversation_template, tokenize_prompt_dataset
from datasets import dataset_dict, load_dataset
from transformers import AutoTokenizer

from colossalai.logging import get_dist_logger

logger = get_dist_logger()


def main():
    parser = argparse.ArgumentParser()
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
        "--conversation_template_config", type=str, default="conversation_template_config", help="Path \
        to save conversation template config files."
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
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    if os.path.exists(args.conversation_template_config):
        conversation_template_config = json.load(open(args.conversation_template_config, "r", encoding='utf8'))
        conversation_template = setup_conversation_template(tokenizer, 
                                chat_template_config=conversation_template_config, 
                                save_path=args.conversation_template_config)
    else:
        chat_template_config = {'system_message':"A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"}  # Use default system message
        conversation_template = setup_conversation_template(tokenizer, chat_template_config=chat_template_config, 
                                save_path=args.conversation_template_config)
    if hasattr(tokenizer, 'pad_token') and hasattr(tokenizer, 'eos_token') and tokenizer.eos_token is not None:
        try:
            # Some tokenizers doesn't allow to set pad_token mannually e.g., Qwen
           tokenizer.pad_token = tokenizer.eos_token
        except AttributeError as e:
            logger.warning(f"Unable to set pad token to eos token, {str(e)}")
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        logger.warning("The tokenizer does not have a pad token which is required. May lead to unintended behavior in training, Please consider manually set them.")

    list_dataset = load_dataset(
        path="json",
        data_files=input_data_paths,
        cache_dir=os.path.join(args.data_cache_dir, "raw"),
        keep_in_memory=False,
        split=train_splits,
        num_proc=cpu_count(),
    )
    for index, dataset in enumerate(list_dataset):
        assert isinstance(dataset, dataset_dict.Dataset)
        if args.num_samples_per_datafile > 0:
            # limit the number of samples in each dataset
            dataset = dataset.select(
                random.sample(range(len(dataset)), min(args.num_samples_per_datafile, len(dataset)))
            )
        logger.info(f"Start to process part-{index}/{len(list_dataset)} of all original datasets.")
        dataset = dataset.map(
            function=tokenize_prompt_dataset,
            fn_kwargs={
                "tokenizer": tokenizer,
                "conversation_template": conversation_template,
                "max_length": args.max_length,
            },
            keep_in_memory=False,
            num_proc=min(len(dataset), cpu_count()),
        )

        dataset = dataset.filter(lambda data: data["input_ids"] is not None)
        dataset = dataset.sort(column_names=("seq_category", "seq_length"), reverse=False, keep_in_memory=False)

        # We don't concatenate data samples here.
        spliced_dataset = dataset
        # Save each jsonl spliced dataset.
        output_index = "0" * (5 - len(str(index))) + str(index)
        output_name = f"part-{output_index}"
        output_jsonl_path = os.path.join(args.data_jsonl_output_dir, output_name + ".jsonl")
        # st = time.time()
        with open(file=output_jsonl_path, mode="w", encoding="utf-8") as fp_writer:
            spliced_count = 0
            for spliced_data_point in spliced_dataset:
                if spliced_count % 500 == 0:
                    logger.info(f"processing {spliced_count} spliced data points for {fp_writer.name}")
                spliced_count += 1
                fp_writer.write(json.dumps(spliced_data_point, ensure_ascii=False) + "\n")

        # Save each arrow spliced dataset
        output_arrow_path = os.path.join(args.data_arrow_output_dir, output_name)
        logger.info(f"Start to save {output_arrow_path}")
        spliced_dataset = load_dataset(
            path="json",
            data_files=[output_jsonl_path],
            cache_dir=os.path.join(args.data_cache_dir, "spliced_and_tokenized"),
            keep_in_memory=False,
            num_proc=cpu_count(),
            split="train",
        )
        spliced_dataset.save_to_disk(dataset_path=output_arrow_path, num_proc=min(len(spliced_dataset), cpu_count()))


if __name__ == "__main__":
    main()
