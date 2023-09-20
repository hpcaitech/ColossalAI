import argparse

__all__ = ["parse_args"]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--distplan",
        type=str,
        default="CAI_Gemini",
        help="The distributed plan [colossalai, zero1, zero2, torch_ddp, torch_zero].",
    )
    parser.add_argument(
        "--tp_degree",
        type=int,
        default=1,
        help="Tensor Parallelism Degree. Valid when using colossalai as dist plan.",
    )
    parser.add_argument(
        "--placement",
        type=str,
        default="cpu",
        help="Placement Policy for Gemini. Valid when using colossalai as dist plan.",
    )
    parser.add_argument(
        "--shardinit",
        action="store_true",
        help="Shard the tensors when init the model to shrink peak memory size on the assigned device. Valid when using colossalai as dist plan.",
    )

    parser.add_argument("--lr", type=float, required=True, help="initial learning rate")
    parser.add_argument("--epoch", type=int, required=True, help="number of epoch")
    parser.add_argument("--data_path_prefix", type=str, required=True, help="location of the train data corpus")
    parser.add_argument(
        "--eval_data_path_prefix", type=str, required=True, help="location of the evaluation data corpus"
    )
    parser.add_argument("--tokenizer_path", type=str, required=True, help="location of the tokenizer")
    parser.add_argument("--max_seq_length", type=int, default=512, help="sequence length")
    parser.add_argument(
        "--refresh_bucket_size",
        type=int,
        default=1,
        help="This param makes sure that a certain task is repeated for this time steps to \
        optimize on the back propagation speed with APEX's DistributedDataParallel",
    )
    parser.add_argument(
        "--max_predictions_per_seq",
        "--max_pred",
        default=80,
        type=int,
        help="The maximum number of masked tokens in a sequence to be predicted.",
    )
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="accumulation_steps")
    parser.add_argument("--train_micro_batch_size_per_gpu", default=2, type=int, required=True, help="train batch size")
    parser.add_argument("--eval_micro_batch_size_per_gpu", default=2, type=int, required=True, help="eval batch size")
    parser.add_argument("--num_workers", default=8, type=int, help="")
    parser.add_argument("--async_worker", action="store_true", help="")
    parser.add_argument("--bert_config", required=True, type=str, help="location of config.json")
    parser.add_argument("--wandb", action="store_true", help="use wandb to watch model")
    parser.add_argument("--wandb_project_name", default="roberta", help="wandb project name")
    parser.add_argument("--log_interval", default=100, type=int, help="report interval")
    parser.add_argument("--log_path", type=str, required=True, help="log file which records train step")
    parser.add_argument("--tensorboard_path", type=str, required=True, help="location of tensorboard file")
    parser.add_argument(
        "--colossal_config", type=str, required=True, help="colossal config, which contains zero config and so on"
    )
    parser.add_argument(
        "--ckpt_path", type=str, required=True, help="location of saving checkpoint, which contains model and optimizer"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--vscode_debug", action="store_true", help="use vscode to debug")
    parser.add_argument("--load_pretrain_model", default="", type=str, help="location of model's checkpoint")
    parser.add_argument(
        "--load_optimizer_lr",
        default="",
        type=str,
        help="location of checkpoint, which contains optimizer, learning rate, epoch, shard and global_step",
    )
    parser.add_argument("--resume_train", action="store_true", help="whether resume training from a early checkpoint")
    parser.add_argument("--mlm", default="bert", type=str, help="model type, bert or deberta")
    parser.add_argument("--checkpoint_activations", action="store_true", help="whether to use gradient checkpointing")

    args = parser.parse_args()
    return args
