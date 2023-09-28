import argparse


def parse_demo_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="google/vit-base-patch16-224",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_path", type=str, default="./output_model", help="The path of your saved model after finetuning."
    )
    parser.add_argument(
        "--plugin",
        type=str,
        default="gemini",
        help="Plugin to use. Valid plugins include 'torch_ddp','torch_ddp_fp16','gemini','low_level_zero', 'hybrid_parallel'.",
    )
    parser.add_argument("--num_epoch", type=int, default=3, help="Number of epochs.")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size (per dp group) for the training dataloader."
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=1,
        help="The size along tensor parallel dimension, only be used when enabling hybrid parallel.",
    )
    parser.add_argument(
        "--pp_size",
        type=int,
        default=1,
        help="The size along pipeline parallel dimension, only be used when enabling hybrid parallel.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.3, help="Ratio of warmup steps against total training steps."
    )
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay to use.")
    parser.add_argument("--grad_checkpoint", type=bool, default=True, help="Whether to use gradient checkpointing.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")

    args = parser.parse_args()
    return args


def parse_benchmark_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="google/vit-base-patch16-224",
        help="Path to a pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--plugin",
        type=str,
        default="gemini",
        help="Plugin to use. Valid plugins include 'torch_ddp','torch_ddp_fp16','gemini','low_level_zero', 'hybrid_parallel'.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size (per dp group) for the training dataloader."
    )
    parser.add_argument("--num_labels", type=int, default=10, help="Number of labels for classification.")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--grad_checkpoint", type=bool, default=True, help="Whether to use gradient checkpointing.")
    parser.add_argument("--max_train_steps", type=int, default=20, help="Total number of training steps to perform.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--mem_cap", type=int, default=0, help="Limit on the usage of space for each GPU (in GB).")
    args = parser.parse_args()

    return args
