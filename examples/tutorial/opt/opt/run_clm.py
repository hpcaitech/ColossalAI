#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import math
import os
import time
from itertools import chain

import datasets
import torch
import torch.distributed as dist
import transformers.utils.logging as logging
from accelerate.utils import set_seed
from context import barrier_context
from datasets import load_dataset
from packaging import version
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoTokenizer,
    GPT2Tokenizer,
    OPTForCausalLM,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils.versions import require_version

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.legacy.context import ParallelMode
from colossalai.legacy.core import global_context as gpc
from colossalai.legacy.tensor import ProcessGroup
from colossalai.legacy.utils import get_dataloader
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import HybridAdam
from colossalai.zero import GeminiOptimizer

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def get_time_stamp():
    torch.cuda.synchronize()
    return time.time()


def parse_args():
    parser = colossalai.legacy.get_default_parser()
    parser.add_argument("-s", "--synthetic", action="store_true")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )

    parser.add_argument("--mem_cap", type=int, default=0, help="use mem cap")
    parser.add_argument("--init_in_cpu", action="store_true", default=False, help="init training model in cpu")
    args = parser.parse_args()

    # Sanity checks
    if not args.synthetic:
        if args.dataset_name is None and args.train_file is None and args.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if args.train_file is not None:
                extension = args.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, json or txt file."
            if args.validation_file is not None:
                extension = args.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, json or txt file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def colo_memory_cap(size_in_GB):
    from colossalai.utils import colo_device_memory_capacity, colo_set_process_memory_fraction

    cuda_capacity = colo_device_memory_capacity(get_accelerator().get_current_device())
    if size_in_GB * (1024**3) < cuda_capacity:
        colo_set_process_memory_fraction(size_in_GB * (1024**3) / cuda_capacity)
        print("Using {} GB of GPU memory".format(size_in_GB))


class DummyDataloader:
    def __init__(self, length, batch_size, seq_len, vocab_size):
        self.length = length
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def generate(self):
        input_ids = torch.randint(
            0, self.vocab_size, (self.batch_size, self.seq_len), device=get_accelerator().get_current_device()
        )
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids}

    def __iter__(self):
        self.step = 0
        return self

    def __next__(self):
        if self.step < self.length:
            self.step += 1
            return self.generate()
        else:
            raise StopIteration

    def __len__(self):
        return self.length


def main():
    args = parse_args()
    disable_existing_loggers()
    colossalai.legacy.launch_from_torch()
    logger = get_dist_logger()
    is_main_process = dist.get_rank() == 0

    if is_main_process:
        datasets.utils.logging.set_verbosity_warning()
        logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        logging.set_verbosity_error()

    if args.mem_cap > 0:
        colo_memory_cap(args.mem_cap)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        logger.info(f"Rank {dist.get_rank()}: random seed is set to {args.seed}")

    # Handle the repository creation
    with barrier_context():
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    logger.info("Start preparing dataset", ranks=[0])
    if not args.synthetic:
        if args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(
                    args.dataset_name,
                    args.dataset_config_name,
                    split=f"train[:{args.validation_split_percentage}%]",
                )
                raw_datasets["train"] = load_dataset(
                    args.dataset_name,
                    args.dataset_config_name,
                    split=f"train[{args.validation_split_percentage}%:]",
                )
        else:
            data_files = {}
            dataset_args = {}
            if args.train_file is not None:
                data_files["train"] = args.train_file
            if args.validation_file is not None:
                data_files["validation"] = args.validation_file
            extension = args.train_file.split(".")[-1]
            if extension == "txt":
                extension = "text"
                dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
            raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
            # If no validation data is there, validation_split_percentage will be used to divide the dataset.
            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[:{args.validation_split_percentage}%]",
                    **dataset_args,
                )
                raw_datasets["train"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[{args.validation_split_percentage}%:]",
                    **dataset_args,
                )
    logger.info("Dataset is prepared", ranks=[0])

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
    logger.info("Model config has been created", ranks=[0])

    if args.model_name_or_path == "facebook/opt-13b":
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    else:
        print(f"load model from {args.model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    logger.info(f"{tokenizer.__class__.__name__} has been created", ranks=[0])

    if args.init_in_cpu:
        init_dev = torch.device("cpu")
    else:
        init_dev = get_accelerator().get_current_device()

    cai_version = colossalai.__version__
    logger.info(f"using Colossal-AI version {cai_version}")
    # build model
    if version.parse(cai_version) >= version.parse("0.3.1"):
        from contextlib import nullcontext

        from colossalai.lazy import LazyInitContext

        ctx = (
            LazyInitContext(default_device=init_dev)
            if args.model_name_or_path is None or args.model_name_or_path == "facebook/opt-13b"
            else nullcontext()
        )
    else:
        from colossalai.zero import ColoInitContext

        ctx = ColoInitContext(device=init_dev)
    if args.model_name_or_path is None or args.model_name_or_path == "facebook/opt-13b":
        # currently, there has a bug in pretrained opt-13b
        # we can not import it until huggingface fix it
        logger.info("Train a new model from scratch", ranks=[0])
        with ctx:
            model = OPTForCausalLM(config)
    else:
        logger.info("Finetune a pre-trained model", ranks=[0])
        with ctx:
            model = OPTForCausalLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                local_files_only=False,
            )

    # enable graident checkpointing
    model.gradient_checkpointing_enable()

    PLACEMENT_POLICY = "auto"
    if version.parse(cai_version) >= version.parse("0.3.1"):
        from colossalai.zero import GeminiDDP

        model = GeminiDDP(model, offload_optim_frac=1.0, pin_memory=True)
    elif version.parse(cai_version) > version.parse("0.1.10"):
        try:
            from colossalai.nn.parallel import GeminiDDP
        except ImportError:
            # this works for unreleased main branch, and this may be released on 0.2.9
            from colossalai.zero import GeminiDDP
        model = GeminiDDP(
            model, device=get_accelerator().get_current_device(), placement_policy=PLACEMENT_POLICY, pin_memory=True
        )
    elif version.parse(cai_version) <= version.parse("0.1.10") and version.parse(cai_version) >= version.parse("0.1.9"):
        from colossalai.gemini import ChunkManager, GeminiManager

        pg = ProcessGroup()
        chunk_size = ChunkManager.search_chunk_size(model, 64 * 1024**2, 32)
        chunk_manager = ChunkManager(
            chunk_size,
            pg,
            enable_distributed_storage=True,
            init_device=GeminiManager.get_default_device(PLACEMENT_POLICY),
        )
        gemini_manager = GeminiManager(PLACEMENT_POLICY, chunk_manager)
        model = ZeroDDP(model, gemini_manager)

    logger.info(f"{model.__class__.__name__} has been created", ranks=[0])

    if not args.synthetic:
        # Preprocessing the datasets.
        # First we tokenize all the texts.
        column_names = raw_datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        def tokenize_function(examples):
            return tokenizer(examples[text_column_name])

        with barrier_context(executor_rank=0, parallel_mode=ParallelMode.DATA):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
        block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    if not args.synthetic:
        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
        # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
        # to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

        with barrier_context(executor_rank=0, parallel_mode=ParallelMode.DATA):
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )

        train_dataset = lm_datasets["train"]
        eval_dataset = lm_datasets["validation"]

        # Log a few random samples from the training set:
        # for index in random.sample(range(len(train_dataset)), 3):
        #     logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

        # DataLoaders creation:
        train_dataloader = get_dataloader(
            train_dataset,
            shuffle=True,
            add_sampler=True,
            collate_fn=default_data_collator,
            batch_size=args.per_device_train_batch_size,
        )
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size
        )
    else:
        train_dataloader = DummyDataloader(
            30, args.per_device_train_batch_size, config.max_position_embeddings, config.vocab_size
        )
        eval_dataloader = DummyDataloader(
            10, args.per_device_train_batch_size, config.max_position_embeddings, config.vocab_size
        )
    logger.info("Dataloaders have been created", ranks=[0])

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = HybridAdam(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    optimizer = GeminiOptimizer(optimizer, model, initial_scale=2**14)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.per_device_train_batch_size * gpc.get_world_size(ParallelMode.DATA)
    num_train_samples = len(train_dataset) if not args.synthetic else 30 * total_batch_size
    num_eval_samples = len(eval_dataset) if not args.synthetic else 10 * total_batch_size

    logger.info("***** Running training *****", ranks=[0])
    logger.info(f"  Num examples = {num_train_samples}", ranks=[0])
    logger.info(f"  Num Epochs = {args.num_train_epochs}", ranks=[0])
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}", ranks=[0])
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}", ranks=[0])
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}", ranks=[0])
    logger.info(f"  Total optimization steps = {args.max_train_steps}", ranks=[0])

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not is_main_process)
    completed_steps = 0
    starting_epoch = 0
    global_step = 0

    for epoch in range(starting_epoch, args.num_train_epochs):
        if completed_steps >= args.max_train_steps:
            break

        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.cuda() for k, v in batch.items()}
            outputs = model(use_cache=False, **batch)
            loss = outputs["loss"]
            optimizer.backward(loss)

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            global_step += 1
            logger.info("Global step {} finished".format(global_step + 1), ranks=[0])

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                batch = {k: v.cuda() for k, v in batch.items()}
                outputs = model(**batch)

        loss = outputs["loss"].unsqueeze(0)
        losses.append(loss)

        losses = torch.cat(losses)
        losses = losses[:num_eval_samples]
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        logger.info(f"Epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}", ranks=[0])

    if args.output_dir is not None:
        model_state = model.state_dict()
        if is_main_process:
            torch.save(model_state, args.output_dir + "/epoch_{}_model.pth".format(completed_steps))
        dist.barrier()
        # load_state = torch.load(args.output_dir + '/epoch_{}_model.pth'.format(completed_steps))
        # model.load_state_dict(load_state, strict=False)

    logger.info("Training finished", ranks=[0])


if __name__ == "__main__":
    main()
