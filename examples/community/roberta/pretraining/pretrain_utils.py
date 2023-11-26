import os
import sys

import torch
import transformers
from transformers import get_linear_schedule_with_warmup

from colossalai.legacy.core import global_context as gpc
from colossalai.nn.optimizer import HybridAdam

sys.path.append(os.getcwd())
from collections import OrderedDict

import torch.nn as nn
from model.bert import BertForMaskedLM
from model.deberta_v2 import DebertaV2ForMaskedLM

__all__ = ["get_model", "get_optimizer", "get_lr_scheduler", "get_dataloader_for_pretraining"]


def get_new_state_dict(state_dict, start_index=13):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[start_index:]
        new_state_dict[name] = v
    return new_state_dict


class LMModel(nn.Module):
    def __init__(self, model, config, args):
        super().__init__()

        self.checkpoint = args.checkpoint_activations
        self.config = config
        self.model = model
        if self.checkpoint:
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        # Only return lm_logits
        return self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)


def get_model(args, logger):
    if args.mlm == "bert":
        config = transformers.BertConfig.from_json_file(args.bert_config)
        model = BertForMaskedLM(config)
    elif args.mlm == "deberta_v2":
        config = transformers.DebertaV2Config.from_json_file(args.bert_config)
        model = DebertaV2ForMaskedLM(config)
    else:
        raise Exception("Invalid mlm!")

    if len(args.load_pretrain_model) > 0:
        assert os.path.exists(args.load_pretrain_model)
        # load_checkpoint(args.load_pretrain_model, model, strict=False)
        m_state_dict = torch.load(
            args.load_pretrain_model, map_location=torch.device(f"cuda:{torch.cuda.current_device()}")
        )
        # new_state_dict = get_new_state_dict(m_state_dict)
        model.load_state_dict(
            m_state_dict, strict=True
        )  # must insure that every process have identical parameters !!!!!!!
        logger.info("load model success")

    numel = sum([p.numel() for p in model.parameters()])
    if args.checkpoint_activations:
        model.gradient_checkpointing_enable()
    # model = LMModel(model, config, args)

    return config, model, numel


def get_optimizer(model, lr):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta", "LayerNorm"]

    # configure the weight decay for bert models
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.1},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = HybridAdam(optimizer_grouped_parameters, lr=lr, betas=[0.9, 0.95])
    return optimizer


def get_lr_scheduler(optimizer, total_steps, warmup_steps=2000, last_epoch=-1):
    # warmup_steps = int(total_steps * warmup_ratio)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps, last_epoch=last_epoch
    )
    # lr_scheduler = LinearWarmupLR(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)
    return lr_scheduler


def save_ckpt(model, optimizer, lr_scheduler, path, epoch, shard, global_step):
    model_path = path + "_pytorch_model.bin"
    optimizer_lr_path = path + ".op_lrs"
    checkpoint = {}
    checkpoint["optimizer"] = optimizer.state_dict()
    checkpoint["lr_scheduler"] = lr_scheduler.state_dict()
    checkpoint["epoch"] = epoch
    checkpoint["shard"] = shard
    checkpoint["global_step"] = global_step
    model_state = model.state_dict()  # each process must run model.state_dict()
    if gpc.get_global_rank() == 0:
        torch.save(checkpoint, optimizer_lr_path)
        torch.save(model_state, model_path)
