import argparse
import json
import os
import time
from multiprocessing import cpu_count

from datasets import load_dataset
from dummy_dataset import DummyLLMDataset

from colossalai.logging import get_dist_logger

logger = get_dist_logger()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        default=None,
        help="The output dir",
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        required=True,
        default=None,
        help="The size of data",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        required=True,
        default=None,
        help="The max length of data",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        required=True,
        default=None,
        help="The type of data, choose one from ['sft', 'prompt', 'preference', 'kto']",
    )
    args = parser.parse_args()
    if args.data_type == "sft":
        dataset = DummyLLMDataset(["input_ids", "attention_mask", "labels"], args.max_length, args.dataset_size)
    elif args.data_type == "prompt":
        # pass PPO dataset is prepared separately
        pass
    elif args.data_type == "preference":
        dataset = DummyLLMDataset(
            ["chosen_input_ids", "chosen_loss_mask", "rejected_input_ids", "rejected_loss_mask"],
            args.max_length,
            args.dataset_size,
        )
    elif args.data_type == "kto":
        dataset = DummyLLMDataset(
            ["prompt", "completion", "label"],
            args.max_length - 512,
            args.dataset_size,
            gen_fn={
                "completion": lambda x: [1] * 512,
                "label": lambda x: x % 2,
            },
        )
    else:
        raise ValueError(f"Unknown data type {args.data_type}")

    # Save each jsonl spliced dataset.
    output_index = "0"
    output_name = f"part-{output_index}"
    os.makedirs(args.data_dir, exist_ok=True)
    output_jsonl_path = os.path.join(args.data_dir, "json")
    output_arrow_path = os.path.join(args.data_dir, "arrow")
    output_cache_path = os.path.join(args.data_dir, "cache")
    os.makedirs(output_jsonl_path, exist_ok=True)
    os.makedirs(output_arrow_path, exist_ok=True)
    output_jsonl_file_path = os.path.join(output_jsonl_path, output_name + ".jsonl")
    st = time.time()
    with open(file=output_jsonl_file_path, mode="w", encoding="utf-8") as fp_writer:
        count = 0
        for i in range(len(dataset)):
            data_point = dataset[i]
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
    output_arrow_file_path = os.path.join(output_arrow_path, output_name)
    logger.info(f"Start to save {output_arrow_file_path}")
    dataset = load_dataset(
        path="json",
        data_files=[output_jsonl_file_path],
        cache_dir=os.path.join(output_cache_path, "tokenized"),
        keep_in_memory=False,
        num_proc=cpu_count(),
        split="train",
    )
    dataset.save_to_disk(dataset_path=output_arrow_file_path, num_proc=min(len(dataset), cpu_count()))
