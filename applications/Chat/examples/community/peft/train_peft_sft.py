import argparse
import os

import loralib as lora
import torch
import torch.distributed as dist
from coati.dataset import DataCollatorForSupervisedDataset, SFTDataset, SupervisedDataset
from coati.models.base import RewardModel
from coati.models.bloom import BLOOMLM
from coati.models.gpt import GPTLM
from coati.models.llama import LlamaLM
from coati.models.opt import OPTLM
from coati.trainer import SFTTrainer
from coati.trainer.strategies import DDPStrategy, GeminiStrategy, LowLevelZeroStrategy
from datasets import load_dataset
from easy_dataset import EasyDataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, BloomTokenizerFast
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer

from colossalai.logging import get_dist_logger
from colossalai.nn.optimizer import HybridAdam
from colossalai.tensor import ColoParameter


def train(args):
    # configure strategy
    if args.strategy == 'ddp':
        strategy = DDPStrategy()
    elif args.strategy == 'colossalai_gemini':
        strategy = GeminiStrategy(placement_policy='cuda')
    elif args.strategy == 'colossalai_zero2':
        strategy = LowLevelZeroStrategy(stage=2, placement_policy='cuda')
    else:
        raise ValueError(f'Unsupported strategy "{args.strategy}"')

    # configure model
    with strategy.model_init_context():
        print('Warning: currently only bloom is tested, gpt2,llama and opt are not tested')
        model = AutoModelForCausalLM.from_pretrained(args.pretrain).to(torch.cuda.current_device())
        # if the args.save_path exists and args.save_path+'/adapter_config.json' exists, we'll load the adapter_config.json
        if os.path.exists(args.save_path) and os.path.exists(args.save_path + '/adapter_config.json') \
                and os.path.exists(args.save_path + '/adapter_model.bin'):
            print("loading from saved peft model ", args.save_path)
            model = PeftModel.from_pretrained(model, args.save_path)
        else:
            # we'll use peft lora library to do the lora
            lora_rank = args.lora_rank if args.lora_rank > 0 else 32
            # config lora with rank of lora_rank
            lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                     inference_mode=False,
                                     r=lora_rank,
                                     lora_alpha=32,
                                     lora_dropout=0.1)
            model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # configure tokenizer
    if args.model == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == 'bloom':
        tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == 'opt':
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == 'llama':
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrain,
            padding_side="right",
            use_fast=False,
        )
        tokenizer.eos_token = '<\s>'
        tokenizer.pad_token = tokenizer.unk_token
    else:
        raise ValueError(f'Unsupported model "{args.model}"')

    if args.model == 'llama' and args.strategy == 'colossalai_gemini':
        # this is a hack to deal with the resized embedding
        # to make sure all parameters are ColoParameter for Colossal-AI Gemini Compatibility
        for name, param in model.named_parameters():
            if not isinstance(param, ColoParameter):
                sub_module_name = '.'.join(name.split('.')[:-1])
                weight_name = name.split('.')[-1]
                sub_module = model.get_submodule(sub_module_name)
                setattr(sub_module, weight_name, ColoParameter(param))

    # configure optimizer
    if args.strategy.startswith('colossalai'):
        optim = HybridAdam(model.parameters(), lr=args.lr, clipping_norm=1.0)
    else:
        optim = Adam(model.parameters(), lr=args.lr)

    logger = get_dist_logger()
    logger.set_level('WARNING')

    # configure dataset
    law_dataset = EasyDataset(args.dataset, tokenizer=tokenizer, is_group_texts=not args.is_short_text)
    train_dataset = law_dataset
    print(train_dataset)
    eval_dataset = None
    if args.eval_dataset is not None:
        eval_dataset = EasyDataset(args.eval_dataset, tokenizer=tokenizer, is_group_texts=not args.is_short_text)
    data_collator = default_collate
    if dist.is_initialized() and dist.get_world_size() > 1:
        train_sampler = DistributedSampler(train_dataset,
                                           shuffle=True,
                                           seed=42,
                                           drop_last=True,
                                           rank=dist.get_rank(),
                                           num_replicas=dist.get_world_size())
        if eval_dataset is not None:
            eval_sampler = DistributedSampler(eval_dataset,
                                              shuffle=False,
                                              seed=42,
                                              drop_last=False,
                                              rank=dist.get_rank(),
                                              num_replicas=dist.get_world_size())
    else:
        train_sampler = None
        eval_sampler = None

    train_dataloader = DataLoader(train_dataset,
                                  shuffle=(train_sampler is None),
                                  sampler=train_sampler,
                                  batch_size=args.batch_size,
                                  collate_fn=data_collator,
                                  pin_memory=True)
    if eval_dataset is not None:
        eval_dataloader = DataLoader(eval_dataset,
                                     shuffle=(eval_sampler is None),
                                     sampler=eval_sampler,
                                     batch_size=args.batch_size,
                                     collate_fn=data_collator,
                                     pin_memory=True)
    else:
        eval_dataloader = None

    trainer = SFTTrainer(model=model,
                         strategy=strategy,
                         optim=optim,
                         train_dataloader=train_dataloader,
                         eval_dataloader=eval_dataloader,
                         batch_size=args.batch_size,
                         max_epochs=args.max_epochs,
                         accumulation_steps=args.accumulation_steps)

    trainer.fit(logger=logger, log_interval=args.log_interval)

    # save model checkpoint after fitting on only rank0
    trainer.save_model(path=args.save_path, only_rank0=True, tokenizer=tokenizer)
    # save optimizer checkpoint on all ranks
    if args.need_optim_ckpt:
        strategy.save_optimizer(trainer.optimizer,
                                'rm_optim_checkpoint_%d.pt' % (torch.cuda.current_device()),
                                only_rank0=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy',
                        choices=['ddp', 'colossalai_gemini', 'colossalai_zero2'],
                        default='ddp')
    parser.add_argument('--model', choices=['gpt2', 'bloom', 'opt', 'llama'], default='bloom')
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--eval_dataset', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='output')
    parser.add_argument('--need_optim_ckpt', type=bool, default=False)
    parser.add_argument('--max_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lora_rank', type=int, default=0, help="low-rank adaptation matrices rank")
    parser.add_argument('--log_interval', type=int, default=100, help="how many steps to log")
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--accumulation_steps', type=int, default=8)
    parser.add_argument('--enable_peft_lora', action='store_true', default=False)
    parser.add_argument("--is_short_text", action='store_true', default=False)
    args = parser.parse_args()
    train(args)
