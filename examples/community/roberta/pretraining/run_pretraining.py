import math
import os
import time
from functools import partial

import torch
from arguments import parse_args
from evaluation import evaluate
from loss import LossForPretraining
from nvidia_bert_dataset_provider import NvidiaBertDatasetProvider
from pretrain_utils import get_lr_scheduler, get_model, get_optimizer, save_ckpt
from tqdm import tqdm
from transformers import AutoTokenizer
from utils.exp_util import get_mem_info, get_tflops, log_args, throughput_calculator
from utils.global_vars import get_tensorboard_writer, get_timers, set_global_variables
from utils.logger import Logger

import colossalai
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.nn.parallel import GeminiDDP, zero_model_wrapper, zero_optim_wrapper
from colossalai.tensor import ColoParameter, ComputePattern, ComputeSpec, ProcessGroup, ReplicaSpec, ShardSpec
from colossalai.utils import get_current_device
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.zero import ZeroOptimizer


def main():

    args = parse_args()
    launch_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    logger = Logger(os.path.join(args.log_path, launch_time), cuda=torch.cuda.is_available(), debug=args.vscode_debug)

    if args.vscode_debug:
        colossalai.launch(config={},
                          rank=args.rank,
                          world_size=args.world_size,
                          host=args.host,
                          port=args.port,
                          backend=args.backend)
        args.local_rank = -1
        args.log_interval = 1
    else:
        colossalai.launch_from_torch(config={})    #args.colossal_config
        args.local_rank = int(os.environ["LOCAL_RANK"])
        logger.info(
            f'launch_from_torch, world size: {torch.distributed.get_world_size()} | ' +
            f'ParallelMode.MODEL: {ParallelMode.MODEL} | ParallelMode.DATA: {ParallelMode.DATA} | ParallelMode.TENSOR: {ParallelMode.TENSOR}'
        )

    log_args(logger, args)
    args.tokenizer = tokenizer
    args.logger = logger
    set_global_variables(launch_time, args.tensorboard_path)

    world_size = torch.distributed.get_world_size()
    init_dev = get_current_device()

    # build model, optimizer and criterion
    if args.distplan.startswith("CAI"):
        # all param must use the same process group.
        world_size = torch.distributed.get_world_size()
        shard_pg = ProcessGroup(tp_degree=world_size) if args.shardinit else None
        default_dist_spec = ShardSpec([-1], [world_size]) if args.shardinit else None

        if args.shardinit and args.distplan != "CAI_Gemini":
            raise RuntimeError("You can only use shardinit with CAI_Gemini")

        # build GPT model
        with ColoInitContext(device=get_current_device(),
                             dtype=torch.half,
                             default_dist_spec=default_dist_spec,
                             default_pg=shard_pg):
            config, model, numel = get_model(args, logger)

        # assign running configurations
        gemini_config = None
        if args.distplan.startswith("CAI_ZeRO"):
            optim_config = dict(reduce_bucket_size=12 * 1024 * 1024, overlap_communication=True, verbose=True)
        elif args.distplan == "CAI_Gemini":
            gemini_config = dict(strict_ddp_mode=args.tp_degree == 1,
                                 device=get_current_device(),
                                 placement_policy=args.placement,
                                 pin_memory=True,
                                 hidden_dim=model.config.hidden_size,
                                 search_range_m=128)
            optim_config = dict(gpu_margin_mem_ratio=0.)
        else:
            raise RuntimeError

        # build a highly optimized gpu/cpu optimizer
        optimizer = get_optimizer(model, lr=args.lr)

        if args.distplan == "CAI_ZeRO1":
            zero_stage = 1
        elif args.distplan == "CAI_ZeRO2":
            zero_stage = 2
        elif args.distplan == "CAI_Gemini":
            zero_stage = 3
        else:
            raise RuntimeError

        # wrap your model and optimizer
        model = zero_model_wrapper(model, zero_stage, gemini_config)
        optimizer = zero_optim_wrapper(model, optimizer, optim_config=optim_config)

        logger.info(get_mem_info(prefix='After init optim, '))

    else:
        config, model, numel = get_model(args, logger)
        logger.info("no_zero")

    if torch.distributed.get_rank() == 0:
        os.mkdir(os.path.join(args.ckpt_path, launch_time))

    logger.info(f'Model numel: {numel}')

    get_tflops_func = partial(get_tflops, numel, args.train_micro_batch_size_per_gpu, args.max_seq_length)

    # 144003367 is is the length of the entire dataset
    steps_per_epoch = 144003367 // world_size // args.train_micro_batch_size_per_gpu // args.gradient_accumulation_steps // args.refresh_bucket_size    #len(dataloader)
    total_steps = steps_per_epoch * args.epoch

    lr_scheduler = get_lr_scheduler(optimizer, total_steps=total_steps, last_epoch=-1)

    start_epoch = 0
    start_shard = 0
    global_step = 0
    if args.resume_train:
        assert os.path.exists(args.load_optimizer_lr)
        o_l_state_dict = torch.load(args.load_optimizer_lr, map_location='cpu')
        o_l_state_dict['lr_scheduler']['last_epoch'] = o_l_state_dict['lr_scheduler']['last_epoch'] - 1
        optimizer.load_state_dict(o_l_state_dict['optimizer'])
        # o_l_state_dict['lr_scheduler']['last_epoch']
        lr_scheduler = get_lr_scheduler(optimizer,
                                        total_steps=total_steps,
                                        last_epoch=o_l_state_dict['lr_scheduler']['last_epoch'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda(f"cuda:{torch.cuda.current_device()}")
        # if you want delete the above three code, must move the model to gpu. Because in optimizer.step()
        lr_scheduler.load_state_dict(o_l_state_dict['lr_scheduler'])

        start_epoch = o_l_state_dict['epoch']
        start_shard = o_l_state_dict['shard'] + 1
        # global_step = o_l_state_dict['global_step'] + 1
        logger.info(
            f'resume from epoch {start_epoch} shard {start_shard} step {lr_scheduler.last_epoch} lr {lr_scheduler.get_last_lr()[0]}'
        )

    criterion = LossForPretraining(config.vocab_size)

    # build dataloader
    pretrain_dataset_provider = NvidiaBertDatasetProvider(args)

    logger.info(get_mem_info(prefix='After init model, '))

    best_loss = None
    eval_loss = 0
    train_loss = 0
    timers = get_timers()
    timers('interval_time').start()
    timers('epoch_time').start()
    timers('shard_time').start()

    for epoch in range(start_epoch, args.epoch):

        for shard in range(start_shard, len(os.listdir(args.data_path_prefix))):

            dataset_iterator, total_length = pretrain_dataset_provider.get_shard(shard)
            # pretrain_dataset_provider.prefetch_shard(shard + 1) # may cause cpu memory overload
            if torch.distributed.get_rank() == 0:
                iterator_data = tqdm(enumerate(dataset_iterator),
                                     total=(total_length // args.train_micro_batch_size_per_gpu // world_size),
                                     colour='cyan',
                                     smoothing=1)
            else:
                iterator_data = enumerate(dataset_iterator)

            model.train()

            for step, batch_data in iterator_data:

                # batch_data = pretrain_dataset_provider.get_batch(batch_index)
                input_ids = batch_data[0].cuda(f"cuda:{torch.cuda.current_device()}")
                attention_mask = batch_data[1].cuda(f"cuda:{torch.cuda.current_device()}")
                token_type_ids = batch_data[2].cuda(f"cuda:{torch.cuda.current_device()}")
                mlm_label = batch_data[3].cuda(f"cuda:{torch.cuda.current_device()}")
                # nsp_label = batch_data[5].cuda()

                output = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

                loss = criterion(output.logits, mlm_label)
                pretrain_dataset_provider.prefetch_batch()

                optimizer.backward(loss)
                train_loss += loss.float().item()
                # if  (step + 1) % args.accumulation_step == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                if global_step % args.log_interval == 0 and global_step != 0 \
                        and torch.distributed.get_rank() == 0:
                    elapsed_time = timers('interval_time').elapsed(reset=False)
                    elapsed_time_per_iteration = elapsed_time / global_step
                    samples_per_sec, tflops, approx_parameters_in_billions = throughput_calculator(
                        numel, args, config, elapsed_time, global_step, world_size)

                    cur_loss = train_loss / args.log_interval
                    current_lr = lr_scheduler.get_last_lr()[0]
                    log_str = f'| epoch: {epoch} | shard: {shard} | step: {global_step} | lr {current_lr:.7f} | elapsed_time: {elapsed_time / 60 :.3f} minutes ' + \
                              f'| mins/batch: {elapsed_time_per_iteration :.3f} seconds | loss: {cur_loss:.7f} | ppl: {math.exp(cur_loss):.3f} | TFLOPS: {get_tflops_func(elapsed_time_per_iteration):.3f} or {tflops:.3f}'
                    logger.info(log_str, print_=False)

                    if args.wandb:
                        tensorboard_log = get_tensorboard_writer()
                        tensorboard_log.log_train(
                            {
                                'lr': current_lr,
                                'loss': cur_loss,
                                'ppl': math.exp(cur_loss),
                                'mins_batch': elapsed_time_per_iteration
                            }, global_step)

                    train_loss = 0

            logger.info(f'epoch {epoch} shard {shard} has cost {timers("shard_time").elapsed() / 60 :.3f} mins')
            logger.info('*' * 100)

            eval_loss += evaluate(model, args, logger, global_step, criterion)
            save_ckpt(model, optimizer, lr_scheduler,
                      os.path.join(args.ckpt_path, launch_time, f'epoch-{epoch}_shard-{shard}_' + launch_time), epoch,
                      shard, global_step)

        eval_loss /= len(os.listdir(args.data_path_prefix))
        logger.info(
            f'epoch {epoch} | shard_length {len(os.listdir(args.data_path_prefix))} | elapsed_time: {timers("epoch_time").elapsed() / 60 :.3f} mins'
            + f'eval_loss: {eval_loss} | ppl: {math.exp(eval_loss)}')
        logger.info('-' * 100)
        if args.wandb and torch.distributed.get_rank() == 0:
            tensorboard_log = get_tensorboard_writer()
            tensorboard_log.log_eval({
                'all_eval_shard_loss': eval_loss,
            }, epoch)
        start_shard = 0
        eval_loss = 0

    pretrain_dataset_provider.release_shard()

    logger.info('Congratulation, training has finished!!!')


if __name__ == '__main__':
    main()
