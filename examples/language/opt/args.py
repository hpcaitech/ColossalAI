import argparse


def parse_demo_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/opt-350m",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_path", type=str, default="./output_model.bin", help="The path of your saved model after finetuning."
    )
    parser.add_argument(
        "--plugin",
        type=str,
        default="gemini",
        help="Plugin to use. Valid plugins include 'torch_ddp','torch_ddp_fp16','gemini','low_level_zero', 'hybrid_parallel'.",
    )
    parser.add_argument("--num_epoch", type=int, default=10, help="Number of epochs.")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size (per dp group) for the training dataloader."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.1, help="Ratio of warmup steps against total training steps."
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")

    args = parser.parse_args()
    return args


def parse_benchmark_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/opt-125m",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--plugin",
        type=str,
        default="gemini",
        help="Plugin to use. Valid plugins include 'torch_ddp','torch_ddp_fp16','gemini','low_level_zero'.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size (per dp group) for the training dataloader."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--max_train_steps", type=int, default=20, help="Total number of training steps to perform.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--mem_cap", type=int, default=0, help="Limit on the usage of space for each GPU (in GB).")
    args = parser.parse_args()

    return args
