import argparse
import warnings

import torch
import torch.distributed as dist
from coati.dataset import HhRlhfDataset, RmStaticDataset
from coati.models.bloom import BLOOMActor
from coati.models.gpt import GPTActor
from coati.models.llama import LlamaActor
from coati.models.opt import OPTActor
from coati.trainer import DPOTrainer
from coati.trainer.strategies import DDPStrategy, GeminiStrategy, LowLevelZeroStrategy
from datasets import load_dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoConfig, AutoTokenizer, BloomTokenizerFast, GPT2Tokenizer, LlamaTokenizer

from colossalai.nn.optimizer import HybridAdam


def main(args):
    # configure strategy
    if args.strategy == "ddp":
        strategy = DDPStrategy()
    elif args.strategy == "colossalai_gemini":
        strategy = GeminiStrategy(placement_policy="static", initial_scale=2**5)
    elif args.strategy == "colossalai_zero2":
        strategy = LowLevelZeroStrategy(stage=2, placement_policy="cuda")
    else:
        raise ValueError(f'Unsupported strategy "{args.strategy}"')

    if args.lora_rank > 0:
        warnings.warn("Lora is not supported yet.")
        args.lora_rank = 0

    with strategy.model_init_context():
        # configure model
        # TODO: add support for llama
        if args.model == "gpt2":
            ref_model = GPTActor(pretrained=args.pretrain)
        elif args.model == "bloom":
            ref_model = BLOOMActor(pretrained=args.pretrain)
        elif args.model == "opt":
            ref_model = OPTActor(pretrained=args.pretrain)
        else:
            raise ValueError(f'Unsupported actor model "{args.model}"')

        ref_model.to(torch.cuda.current_device())

        if args.model == "gpt2":
            config = AutoConfig.from_pretrained(args.pretrain)
            config.embd_pdrop = 0.0
            config.attn_pdrop = 0.0
            config.resid_pdrop = 0.0
            actor = GPTActor(pretrained=args.pretrain, config=config, lora_rank=args.lora_rank)
        elif args.model == "bloom":
            config = AutoConfig.from_pretrained(args.pretrain)
            # TODO: find a proper hyperparameter setting for BLOOM
            config.attention_dropout = 0.0001
            config.hidden_dropout = 0.0001
            actor = BLOOMActor(pretrained=args.pretrain, config=config, lora_rank=args.lora_rank)
        elif args.model == "opt":
            config = AutoConfig.from_pretrained(args.pretrain)
            # TODO: find a proper hyperparameter setting for OPT
            config.attention_dropout = 0.0001
            config.dropout = 0.0001
            config.layerdrop = 0.000
            actor = OPTActor(pretrained=args.pretrain, config=config, lora_rank=args.lora_rank)
        elif args.model == "llama":
            # Note: llama disable dropout by default
            actor = LlamaActor(pretrained=args.pretrain, config=config, lora_rank=args.lora_rank)
        else:
            raise ValueError(f'Unsupported actor model "{args.model}"')

        actor.to(torch.cuda.current_device())

    # configure optimizer
    if args.strategy.startswith("colossalai"):
        actor_optim = HybridAdam(actor.parameters(), lr=args.lr)
    else:
        actor_optim = Adam(actor.parameters(), lr=args.lr)

    # configure tokenizer
    if args.model == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2" if args.tokenizer is None else args.tokenizer)
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == "bloom":
        tokenizer = BloomTokenizerFast.from_pretrained(
            "bigscience/bloom-560m" if args.tokenizer is None else args.tokenizer
        )
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == "opt":
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m" if args.tokenizer is None else args.tokenizer)
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == "llama":
        tokenizer = LlamaTokenizer.from_pretrained(
            "hf-internal-testing/llama-tokenizer" if args.tokenizer is None else args.tokenizer
        )
        tokenizer.eos_token = "<\s>"
        tokenizer.pad_token = tokenizer.unk_token
    else:
        raise ValueError(f'Unsupported model "{args.model}"')
    # NOTE: generate() requires padding_side to be "left"
    tokenizer.padding_side = "right"

    # prepare for data and dataset
    if args.subset is not None:
        data = load_dataset(args.dataset, data_dir=args.subset)
    else:
        data = load_dataset(args.dataset)

    train_data = data["train"].select(range(min(args.max_datasets_size, len(data["train"]))))
    eval_data = data["test"].select(range(min(args.max_datasets_size, len(data["test"]))))

    if args.dataset == "Dahoas/rm-static":
        train_dataset = RmStaticDataset(train_data, tokenizer, args.max_len)
        eval_dataset = RmStaticDataset(eval_data, tokenizer, args.max_len)
    elif args.dataset == "Anthropic/hh-rlhf":
        train_dataset = HhRlhfDataset(train_data, tokenizer, args.max_len)
        eval_dataset = HhRlhfDataset(eval_data, tokenizer, args.max_len)
    else:
        raise ValueError(f'Unsupported dataset "{args.dataset}"')

    if dist.is_initialized() and dist.get_world_size() > 1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=42, drop_last=True)
        eval_sampler = DistributedSampler(eval_dataset, shuffle=False, seed=42, drop_last=False)
    else:
        train_sampler = None
        eval_sampler = None
    train_dataloader = DataLoader(
        train_dataset, shuffle=(train_sampler is None), sampler=train_sampler, batch_size=args.batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, shuffle=(eval_sampler is None), sampler=eval_sampler, batch_size=args.batch_size
    )

    # NOTE: For small models like opt-1.3b, reward model and initial model are not required to be parallelized.
    ref_model = strategy.prepare(ref_model)

    lr_scheduler = CosineAnnealingLR(actor_optim, args.max_epoch * len(train_dataset), eta_min=1e-8)
    strategy_dict = strategy.prepare(dict(model=actor, optimizer=actor_optim, lr_scheduler=lr_scheduler))
    actor = strategy_dict["model"]
    actor_optim = strategy_dict["optimizer"]
    actor_lr_scheduler = strategy_dict["lr_scheduler"]

    """
    strategy: Strategy,
    actor: Actor,
    ref_model: Actor,
    actor_optim: Optimizer,
    actor_lr_scheduler: _LRScheduler,
    tokenizer: PreTrainedTokenizerBase,
    max_epoch: int = 1,
    beta: float = 0.1,
    disable_reference: bool = False
    """
    # configure trainer
    trainer = DPOTrainer(
        strategy,
        actor,
        ref_model,
        actor_optim,
        actor_lr_scheduler,
        tokenizer=tokenizer,
        max_epochs=args.max_epoch,
        beta=args.beta,
        disable_reference=args.disable_reference,
    )

    trainer.fit(
        train_preference_dataloader=train_dataloader,
        eval_preference_dataloader=eval_dataloader,
        log_dir=args.log_dir,
        use_wandb=args.use_wandb,
    )

    if args.lora_rank > 0 and args.merge_lora_weights:
        from coati.models.lora import LORA_MANAGER

        # NOTE: set model to eval to merge LoRA weights
        LORA_MANAGER.merge_weights = True
        actor.eval()
    # save model checkpoint after fitting
    strategy.save_pretrained(actor, path=args.save_path)
    # save optimizer checkpoint on all ranks
    if args.need_optim_ckpt:
        strategy.save_optimizer(
            actor_optim, "actor_optim_checkpoint_prompts_%d.pt" % (torch.cuda.current_device()), only_rank0=False
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None, help="path to the prompt dataset")
    parser.add_argument("--max_datasets_size", type=int, default=50000)
    parser.add_argument(
        "--strategy",
        choices=["ddp", "colossalai_gemini", "colossalai_zero2"],
        default="colossalai_zero2",
        help="strategy to use",
    )
    parser.add_argument("--subset", type=lambda x: None if x == "None" else x, default=None)
    parser.add_argument("--model", default="gpt2", choices=["gpt2", "bloom", "opt", "llama"])
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="actor_checkpoint_prompts")
    parser.add_argument("--need_optim_ckpt", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epoch", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=700)
    parser.add_argument("--lora_rank", type=int, default=0, help="low-rank adaptation matrices rank")
    parser.add_argument("--merge_lora_weights", type=bool, default=True)
    parser.add_argument("--disable_reference", type=bool, default=False)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--log_dir", default="logs", type=str)
    parser.add_argument("--use_wandb", default=False, action="store_true")
    args = parser.parse_args()
    main(args)
