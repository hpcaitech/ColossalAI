import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import _create_vision_transformer
from titans.dataloader.imagenet import build_dali_imagenet
from tqdm import tqdm

import colossalai
from colossalai.core import global_context as gpc
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn import CrossEntropyLoss
from colossalai.nn._ops import *
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.nn.optimizer import HybridAdam
from colossalai.nn.parallel.data_parallel import ColoDDP
from colossalai.tensor import ComputePattern, ComputeSpec, DistSpecManager, ProcessGroup, ShardSpec
from colossalai.utils import get_current_device
from colossalai.utils.model.colo_init_context import ColoInitContext


def init_1d_row_for_linear_weight_spec(model, world_size: int):
    pg = ProcessGroup(tp_degree=world_size)
    spec = (ShardSpec([-1], [pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))
    with DistSpecManager.no_grad():
        for n, p in model.named_parameters():
            if 'weight' in n and 'norm' not in n and 'patch_embed.proj.weight' not in n:
                p.set_process_group(pg)
                p.set_tensor_spec(*spec)


# Similarly, it's col split for Linear but row split for others.
def init_1d_col_for_linear_weight_bias_spec(model, world_size: int):
    pg = ProcessGroup(tp_degree=world_size)
    spec = (ShardSpec([0], [pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))
    with DistSpecManager.no_grad():
        for n, p in model.named_parameters():
            if ('weight' in n or 'bias' in n) and 'norm' not in n and ('patch_embed.proj.weight' not in n
                                                                       and 'patch_embed.proj.bias' not in n):
                p.set_process_group(pg)
                p.set_tensor_spec(*spec)


def init_spec_func(model, tp_type):
    world_size = torch.distributed.get_world_size()
    if tp_type == 'row':
        init_1d_row_for_linear_weight_spec(model, world_size)
    elif tp_type == 'col':
        init_1d_col_for_linear_weight_bias_spec(model, world_size)
    else:
        raise NotImplemented


def train_imagenet():

    parser = colossalai.get_default_parser()
    parser.add_argument('--from_torch', default=True, action='store_true')
    parser.add_argument('--resume_from', default=False)

    args = parser.parse_args()
    colossalai.launch_from_torch(config=args.config)
    use_ddp = gpc.config.USE_DDP

    disable_existing_loggers()

    logger = get_dist_logger()
    if hasattr(gpc.config, 'LOG_PATH'):
        if gpc.get_global_rank() == 0:
            log_path = gpc.config.LOG_PATH
            if not os.path.exists(log_path):
                os.mkdir(log_path)
            logger.log_to_file(log_path)

    logger.info('Build data loader', ranks=[0])
    root = os.environ['DATA']
    train_dataloader, test_dataloader = build_dali_imagenet(root,
                                                            train_batch_size=gpc.config.BATCH_SIZE,
                                                            test_batch_size=gpc.config.BATCH_SIZE)

    logger.info('Build model', ranks=[0])

    model_kwargs = dict(img_size=gpc.config.IMG_SIZE,
                        patch_size=gpc.config.PATCH_SIZE,
                        embed_dim=gpc.config.HIDDEN_SIZE,
                        depth=gpc.config.DEPTH,
                        num_heads=gpc.config.NUM_HEADS,
                        mlp_ratio=gpc.config.MLP_RATIO,
                        num_classes=gpc.config.NUM_CLASSES,
                        drop_rate=0.1,
                        attn_drop_rate=0.1,
                        weight_init='jax')

    with ColoInitContext(device=get_current_device()):
        model = _create_vision_transformer('vit_small_patch16_224', pretrained=False, **model_kwargs)
    init_spec_func(model, gpc.config.TP_TYPE)

    world_size = torch.distributed.get_world_size()
    model = ColoDDP(module=model, process_group=ProcessGroup(tp_degree=world_size))
    logger.info('Build criterion, optimizer, lr_scheduler', ranks=[0])
    optimizer = HybridAdam(model.parameters(), lr=gpc.config.LEARNING_RATE, weight_decay=gpc.config.WEIGHT_DECAY)

    criterion = CrossEntropyLoss()
    lr_scheduler = CosineAnnealingWarmupLR(optimizer=optimizer,
                                           total_steps=gpc.config.NUM_EPOCHS,
                                           warmup_steps=gpc.config.WARMUP_EPOCHS)

    start_epoch = 0
    if args.resume_from:
        load_model = torch.load(args.resume_from + '_model.pth')
        start_epoch = load_model['epoch']
        model.load_state_dict(load_model['model'])
        load_optim = torch.load(args.resume_from + '_optim_rank_{}.pth'.format(dist.get_rank()))
        optimizer.load_state_dict(load_optim['optim'])

    for epoch in range(start_epoch, gpc.config.NUM_EPOCHS):
        model.train()
        for index, (x, y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False):
            x, y = x.cuda(), y.cuda()
            output = model(x)
            loss = criterion(output, y)
            loss = loss / gpc.config.gradient_accumulation
            if use_ddp:
                model.backward(loss)
            else:
                loss.backward()
            if (index + 1) % gpc.config.gradient_accumulation == 0:
                optimizer.step()
                if use_ddp:
                    model.zero_grad()
                else:
                    optimizer.zero_grad()

        logger.info(
            f"Finish Train Epoch [{epoch+1}/{gpc.config.NUM_EPOCHS}] loss: {loss.item():.3f} lr: {optimizer.state_dict()['param_groups'][0]['lr']}",
            ranks=[0])

        model.eval()
        test_loss = 0
        correct = 0
        test_sum = 0
        with torch.no_grad():
            for index, (x, y) in tqdm(enumerate(test_dataloader), total=len(test_dataloader), leave=False):
                x, y = x.cuda(), y.cuda()
                output = model(x)
                test_loss += F.cross_entropy(output, y, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(y.view_as(pred)).sum().item()
                test_sum += y.size(0)

        test_loss /= test_sum
        logger.info(
            f"Finish Test Epoch [{epoch+1}/{gpc.config.NUM_EPOCHS}] loss: {test_loss:.3f} Accuracy: [{correct}/{test_sum}]({correct/test_sum:.3f})",
            ranks=[0])

        lr_scheduler.step()


if __name__ == '__main__':
    train_imagenet()
