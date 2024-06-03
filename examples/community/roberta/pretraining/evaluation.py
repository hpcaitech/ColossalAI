import math
import os

import torch
from nvidia_bert_dataset_provider import NvidiaBertDatasetProvider
from tqdm import tqdm
from utils.global_vars import get_tensorboard_writer, get_timers


def evaluate(model, args, logger, global_step, criterion):
    evaluate_dataset_provider = NvidiaBertDatasetProvider(args, evaluate=True)
    start_shard = 0

    model.eval()
    timers = get_timers()
    eval_step = 0
    eval_loss = 0
    cur_loss = 0
    world_size = torch.distributed.get_world_size()

    with torch.no_grad():
        for shard in range(start_shard, len(os.listdir(args.eval_data_path_prefix))):
            timers("eval_shard_time").start()

            dataset_iterator, total_length = evaluate_dataset_provider.get_shard(shard)
            # evaluate_dataset_provider.prefetch_shard(shard + 1)
            if torch.distributed.get_rank() == 0:
                iterator_data = tqdm(
                    enumerate(dataset_iterator),
                    total=(total_length // args.eval_micro_batch_size_per_gpu // world_size),
                    colour="MAGENTA",
                    smoothing=1,
                )
            else:
                iterator_data = enumerate(dataset_iterator)

            for (
                step,
                batch_data,
            ) in (
                iterator_data
            ):  # tqdm(enumerate(dataset_iterator), total=(total_length // args.train_micro_batch_size_per_gpu // world_size), colour='cyan', smoothing=1):
                # batch_data = pretrain_dataset_provider.get_batch(batch_index)
                eval_step += 1
                input_ids = batch_data[0].cuda()
                attention_mask = batch_data[1].cuda()
                token_type_ids = batch_data[2].cuda()
                mlm_label = batch_data[3].cuda()
                # nsp_label = batch_data[5].cuda()

                output = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

                loss = criterion(output.logits, mlm_label)  # prediction_scores
                evaluate_dataset_provider.prefetch_batch()

                eval_loss += loss.float().item()

            cur_loss = eval_loss / eval_step
            elapsed_time = timers("eval_shard_time").elapsed()
            elapsed_time_per_iteration = elapsed_time / eval_step
            ppl = math.exp(cur_loss)

            if args.wandb and torch.distributed.get_rank() == 0:
                tensorboard_log = get_tensorboard_writer()
                tensorboard_log.log_eval(
                    {"loss": cur_loss, "ppl": ppl, "mins_batch": elapsed_time_per_iteration}, global_step
                )

            eval_log_str = (
                f"evaluation shard: {shard} | step: {eval_step} | elapsed_time: {elapsed_time / 60 :.3f} minutes "
                + f"| mins/batch: {elapsed_time_per_iteration :.3f} seconds | loss: {cur_loss:.7f} | ppl: {ppl:.7f}"
            )

            logger.info(eval_log_str)
            logger.info("-" * 100)
            logger.info("")

    evaluate_dataset_provider.release_shard()
    model.train()
    return cur_loss
