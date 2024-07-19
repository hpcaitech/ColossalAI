#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare sft dataset for fine-tuning
"""

import argparse
import json
import math
import os
from multiprocessing import cpu_count

from colossal_llama.dataset.conversation import LLaMA2_Conv, LLaMA3_Conv
from colossal_llama.dataset.spliced_and_tokenized_dataset import supervised_tokenize_sft
from datasets import dataset_dict, load_dataset
from transformers import AddedToken, AutoTokenizer

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
    parser.add_argument("--data_output_dirs", type=str, default="data_output_dirs", help="Data output directory")
    parser.add_argument("--max_length", type=int, default=8192, help="Max length of each spliced tokenized sequence")
    parser.add_argument("--num_spliced_dataset_bins", type=int, default=10, help="Number of spliced dataset bins")
    parser.add_argument("--llama_version", type=int, default=3, help="LLaMA version")
    args = parser.parse_args()

    if args.num_spliced_dataset_bins >= 100000:
        raise ValueError("Too many spliced divisions, must be smaller than 100000")

    args.data_cache_dir = os.path.join(args.data_output_dirs, "cache")
    args.data_jsonl_output_dir = os.path.join(args.data_output_dirs, "jsonl")
    args.data_arrow_output_dir = os.path.join(args.data_output_dirs, "arrow")

    if not os.path.exists(args.data_cache_dir):
        os.makedirs(args.data_cache_dir)
    if not os.path.exists(args.data_jsonl_output_dir):
        os.makedirs(args.data_jsonl_output_dir)
    if not os.path.exists(args.data_arrow_output_dir):
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

    # Prepare to the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    default_conversation = LLaMA3_Conv

    # Fix </s> split issue: https://github.com/huggingface/transformers/issues/23833
    if args.llama_version == 2:
        tokenizer.add_tokens(AddedToken("</s>", normalized=False, special=True), special_tokens=True)
        default_conversation = LLaMA2_Conv

    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.unk_token = tokenizer.eos_token

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
        logger.info(f"Start to process part-{index}/{len(list_dataset)} of all original datasets.")
        dataset = dataset.map(
            function=supervised_tokenize_sft,
            fn_kwargs={
                "tokenizer": tokenizer,
                "conversation_template": default_conversation,
                "max_length": args.max_length,
            },
            keep_in_memory=False,
            num_proc=min(len(dataset), cpu_count()),
        )

        dataset = dataset.filter(lambda data: data["labels"] is not None)
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
