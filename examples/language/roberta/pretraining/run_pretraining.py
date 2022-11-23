import colossalai
import math
import torch
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
import colossalai.nn as col_nn
from arguments import parse_args
from pretrain_utils import get_model, get_optimizer, get_lr_scheduler, save_ckpt
from utils.exp_util import get_tflops, get_mem_info, throughput_calculator, log_args
from utils.global_vars import set_global_variables, get_timers, get_tensorboard_writer
from utils.logger import Logger
from evaluation import evaluate
from loss import LossForPretraining

from colossalai.zero.init_ctx import ZeroInitContext
from colossalai.zero.shard_utils import TensorShardStrategy
from colossalai.zero.sharded_model import ShardedModelV2
from colossalai.zero.sharded_optim import ShardedOptimizerV2
from nvidia_bert_dataset_provider import NvidiaBertDatasetProvider
from tqdm import tqdm
import os
import time
from functools import partial

from transformers import AutoTokenizer

from colossalai.gemini import ChunkManager, GeminiManager
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.utils import get_current_device
from colossalai.nn.parallel import ZeroDDP
from colossalai.zero import ZeroOptimizer
from colossalai.tensor import ProcessGroup
from colossalai.nn.optimizer import HybridAdam


def main():

    args = parse_args()
    launch_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
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
        colossalai.launch_from_torch(args.colossal_config) #args.colossal_config
        args.local_rank = int(os.environ["LOCAL_RANK"])
        logger.info(f'launch_from_torch, world size: {torch.distributed.get_world_size()} | ' + 
                    f'ParallelMode.MODEL: {ParallelMode.MODEL} | ParallelMode.DATA: {ParallelMode.DATA} | ParallelMode.TENSOR: {ParallelMode.TENSOR}')

    log_args(logger, args)
    args.tokenizer = tokenizer
    args.logger = logger
    set_global_variables(launch_time, args.tensorboard_path)
    
    use_zero = hasattr(gpc.config, 'zero')
    world_size = torch.distributed.get_world_size()

    # build model, optimizer and criterion
    if use_zero:
        shard_strategy = TensorShardStrategy()
        with ZeroInitContext(target_device=torch.cuda.current_device(), shard_strategy=shard_strategy,
                            shard_param=True):
            
            config, model, numel = get_model(args, logger)
            # model = ShardedModelV2(model, shard_strategy, tensor_placement_policy='cpu', reuse_fp16_shard=True)
    else:
        config, model, numel = get_model(args, logger)
        logger.info("no_zero")
    if torch.distributed.get_rank() == 0:
        os.mkdir(os.path.join(args.ckpt_path, launch_time))

    logger.info(f'Model numel: {numel}')
    
    get_tflops_func = partial(get_tflops, numel, args.train_micro_batch_size_per_gpu, args.max_seq_length)
    steps_per_epoch = 144003367 // world_size // args.train_micro_batch_size_per_gpu // args.gradient_accumulation_steps // args.refresh_bucket_size #len(dataloader)
    total_steps = steps_per_epoch * args.epoch

    # build optimizer and lr_scheduler

    start_epoch = 0
    start_shard = 0
    global_step = 0
    if args.resume_train:
        assert os.path.exists(args.load_optimizer_lr)
        o_l_state_dict = torch.load(args.load_optimizer_lr, map_location='cpu')
        o_l_state_dict['lr_scheduler']['last_epoch'] = o_l_state_dict['lr_scheduler']['last_epoch'] - 1
        optimizer = get_optimizer(model, lr=args.lr)
        optimizer.load_state_dict(o_l_state_dict['optimizer'])
        lr_scheduler = get_lr_scheduler(optimizer, total_steps=total_steps, last_epoch=o_l_state_dict['lr_scheduler']['last_epoch']) #o_l_state_dict['lr_scheduler']['last_epoch']
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda(f"cuda:{torch.cuda.current_device()}")
        # if you want delete the above three code, have to move the model to gpu, because in optimizer.step()
        lr_scheduler.load_state_dict(o_l_state_dict['lr_scheduler'])
        
        start_epoch = o_l_state_dict['epoch']
        start_shard = o_l_state_dict['shard'] + 1
        # global_step = o_l_state_dict['global_step'] + 1
        logger.info(f'resume from epoch {start_epoch} shard {start_shard} step {lr_scheduler.last_epoch} lr {lr_scheduler.get_last_lr()[0]}')
    else:
        optimizer = get_optimizer(model, lr=args.lr)
        lr_scheduler = get_lr_scheduler(optimizer, total_steps=total_steps, last_epoch=-1)

    # optimizer = gpc.config.optimizer.pop('type')(
    # model.parameters(), **gpc.config.optimizer)
    # optimizer = ShardedOptimizerV2(model, optimizer, initial_scale=2**5)
    criterion = LossForPretraining(config.vocab_size)

    # build dataloader
    pretrain_dataset_provider = NvidiaBertDatasetProvider(args)

    # initialize with colossalai
    engine, _, _, lr_scheduelr = colossalai.initialize(model=model,
                                optimizer=optimizer,
                                criterion=criterion,
                                lr_scheduler=lr_scheduler)
    
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
                iterator_data = tqdm(enumerate(dataset_iterator), total=(total_length // args.train_micro_batch_size_per_gpu // world_size), colour='cyan', smoothing=1)
            else:
                iterator_data = enumerate(dataset_iterator)

            engine.train()
            
            for step, batch_data in iterator_data: 

                # batch_data = pretrain_dataset_provider.get_batch(batch_index)
                input_ids = batch_data[0].cuda(f"cuda:{torch.cuda.current_device()}")
                attention_mask = batch_data[1].cuda(f"cuda:{torch.cuda.current_device()}")
                token_type_ids = batch_data[2].cuda(f"cuda:{torch.cuda.current_device()}")
                mlm_label = batch_data[3].cuda(f"cuda:{torch.cuda.current_device()}")
                # nsp_label = batch_data[5].cuda()

                output = engine(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                
                loss = engine.criterion(output.logits, mlm_label)
                pretrain_dataset_provider.prefetch_batch()

                engine.backward(loss)
                train_loss += loss.float().item()
                # if  (step + 1) % args.accumulation_step == 0:
                engine.step()
                lr_scheduelr.step()
                engine.zero_grad()
                
                global_step += 1

                if global_step % args.log_interval == 0 and global_step != 0 \
                    and torch.distributed.get_rank() == 0:
                    elapsed_time = timers('interval_time').elapsed(reset=False)
                    elapsed_time_per_iteration = elapsed_time / global_step
                    samples_per_sec, tflops, approx_parameters_in_billions = throughput_calculator(numel, args, config, elapsed_time, global_step, world_size)

                    cur_loss = train_loss / args.log_interval
                    current_lr = lr_scheduelr.get_last_lr()[0]
                    log_str = f'| epoch: {epoch} | shard: {shard} | step: {global_step} | lr {current_lr:.7f} | elapsed_time: {elapsed_time / 60 :.3f} minutes ' + \
                              f'| mins/batch: {elapsed_time_per_iteration :.3f} seconds | loss: {cur_loss:.7f} | ppl: {math.exp(cur_loss):.3f} | TFLOPS: {get_tflops_func(elapsed_time_per_iteration):.3f} or {tflops:.3f}'
                    logger.info(log_str, print_=False)

                    if args.wandb:
                        tensorboard_log = get_tensorboard_writer()
                        tensorboard_log.log_train({
                            'lr': current_lr,
                            'loss': cur_loss,
                            'ppl': math.exp(cur_loss),
                            'mins_batch': elapsed_time_per_iteration
                        }, global_step)

                    train_loss = 0

            logger.info(f'epoch {epoch} shard {shard} has cost {timers("shard_time").elapsed() / 60 :.3f} mins')
            logger.info('*' * 100)

            eval_loss += evaluate(engine, args, logger, global_step)
            save_ckpt(engine.model, optimizer, lr_scheduelr, os.path.join(args.ckpt_path, launch_time, f'epoch-{epoch}_shard-{shard}_' + launch_time), epoch, shard, global_step)
        
        
        eval_loss /= len(os.listdir(args.data_path_prefix))
        logger.info(f'epoch {epoch} | shard_length {len(os.listdir(args.data_path_prefix))} | elapsed_time: {timers("epoch_time").elapsed() / 60 :.3f} mins' + \
                    f'eval_loss: {eval_loss} | ppl: {math.exp(eval_loss)}')
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
