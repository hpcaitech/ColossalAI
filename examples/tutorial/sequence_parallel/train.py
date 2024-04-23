import argparse

import torch
from data.bert_helper import SequenceParallelDataIterator, get_batch_for_sequence_parallel
from data.dummy_dataloader import DummyDataloader
from loss_func.bert_loss import BertLoss
from lr_scheduler import AnnealingLR
from model.bert import BertForPretrain, build_pipeline_bert

import colossalai
from colossalai.legacy.amp import AMP_TYPE
from colossalai.legacy.context.parallel_mode import ParallelMode
from colossalai.legacy.core import global_context as gpc
from colossalai.legacy.utils import is_using_pp
from colossalai.logging import get_dist_logger
from colossalai.nn.layer.layernorm import MixedFusedLayerNorm as LayerNorm
from colossalai.nn.optimizer import FusedAdam
from colossalai.utils import MultiTimer


def process_batch_data(batch_data):
    tokens, types, sentence_order, loss_mask, lm_labels, padding_mask = batch_data
    if gpc.is_first_rank(ParallelMode.PIPELINE):
        data = dict(input_ids=tokens, attention_masks=padding_mask, tokentype_ids=types, lm_labels=lm_labels)
    else:
        data = dict(attention_masks=padding_mask, tokentype_ids=types, lm_labels=lm_labels)
    label = dict(loss_mask=loss_mask, sentence_order=sentence_order)
    return data, label


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--synthetic", action="store_true", help="whether use synthetic data")
    return parser.parse_args()


def pipeline_data_process_func(stage_output, micro_batch_data):
    tokens, types, sentence_order, loss_mask, lm_labels, padding_mask = micro_batch_data
    if gpc.is_first_rank(ParallelMode.PIPELINE):
        data = (tokens, padding_mask, types, lm_labels)
        label = (loss_mask, sentence_order)
    else:
        data = (stage_output, padding_mask, types, lm_labels)
        label = (loss_mask, sentence_order)
    return data, label


def main():
    # initialize
    parse_args()
    colossalai.legacy.launch_from_torch(config="./config.py", seed=1234, backend="nccl")

    logger = get_dist_logger()

    # build synthetic dataloader
    BATCH_SIZE_PER_GPUS = gpc.config.GLOBAL_BATCH_SIZE // gpc.get_world_size(ParallelMode.DATA)
    VOCAB_SIZE = 30528
    trainloader = DummyDataloader(
        batch_size=BATCH_SIZE_PER_GPUS, vocab_size=VOCAB_SIZE, seq_length=gpc.config.SEQ_LENGTH
    )
    validloader = DummyDataloader(
        batch_size=BATCH_SIZE_PER_GPUS, vocab_size=VOCAB_SIZE, seq_length=gpc.config.SEQ_LENGTH
    )

    logger.info("Dataloaders are built", ranks=[0])

    # build model
    if hasattr(gpc.config, "fp16") and gpc.config.fp16.get("mode") == AMP_TYPE.NAIVE:
        is_naive_fp16 = True
    else:
        is_naive_fp16 = False

    use_pipeline = is_using_pp()
    kwargs = dict(
        vocab_size=VOCAB_SIZE,
        hidden_size=gpc.config.HIDDEN_SIZE,
        max_sequence_length=gpc.config.SEQ_LENGTH,
        num_attention_heads=gpc.config.NUM_ATTENTION_HEADS,
        convert_fp16_to_fp32_in_softmax=True,
        is_naive_fp16=is_naive_fp16,
        add_binary_head=gpc.config.ADD_BINARY_HEAD,
    )

    if use_pipeline:
        model = build_pipeline_bert(num_layers=gpc.config.DEPTH, num_chunks=1, **kwargs)
    else:
        model = BertForPretrain(num_layers=gpc.config.DEPTH, **kwargs)

    model = model.half()
    model.reset_parameters()
    logger.info(f"Model is built with softmax in fp32 = {is_naive_fp16}", ranks=[0])

    total_numel = 0
    for p in model.parameters():
        total_numel += p.numel()
    logger.info(f"This model has {total_numel} parameters")

    # build criterion
    criterion = BertLoss()
    logger.info("Criterion is built", ranks=[0])

    # layernorm and bias has no weight decay
    weight_decay_params = {"params": []}
    no_weight_decay_params = {"params": [], "weight_decay": 0.0}
    for module_ in model.modules():
        if isinstance(module_, LayerNorm):
            no_weight_decay_params["params"].extend([p for p in list(module_._parameters.values()) if p is not None])
        else:
            weight_decay_params["params"].extend(
                [p for n, p in list(module_._parameters.items()) if p is not None and n != "bias"]
            )
            no_weight_decay_params["params"].extend(
                [p for n, p in list(module_._parameters.items()) if p is not None and n == "bias"]
            )

    logger.info(
        f"without weight decay param: {len(no_weight_decay_params['params'])}, with weight decay param: {len(weight_decay_params['params'])}"
    )
    # optimizer
    optimizer = FusedAdam(
        (weight_decay_params, no_weight_decay_params), lr=gpc.config.LR, weight_decay=gpc.config.WEIGHT_DECAY
    )
    logger.info("Optimizer is built", ranks=[0])

    # lr scheduler
    # follow Megatron-LM setting
    warmup_steps = int(gpc.config.DECAY_ITERS * gpc.config.WARMUP_FRACTION)
    lr_scheduler = AnnealingLR(
        optimizer=optimizer,
        max_lr=gpc.config.LR,
        min_lr=gpc.config.MIN_LR,
        warmup_steps=warmup_steps,
        decay_steps=gpc.config.DECAY_ITERS,
        decay_style="linear",
    )
    logger.info(f"LR Scheduler is built with {warmup_steps} warmup steps and {gpc.config.DECAY_ITERS} decay steps")

    # # init
    engine, *dummy = colossalai.legacy.initialize(model, optimizer, criterion, verbose=True)

    # build timer
    timer = MultiTimer()

    # build loss tracker
    accumulated_train_loss = torch.zeros(1, dtype=torch.float32).cuda()
    accumulated_eval_loss = torch.zeros(1, dtype=torch.float32).cuda()

    # build data iters for pipeline parallel
    if use_pipeline:
        train_data_iter = SequenceParallelDataIterator(trainloader)
        valid_data_iter = SequenceParallelDataIterator(validloader)
        engine.schedule.data_process_func = pipeline_data_process_func

    logger.info("start training")

    for step in range(1, gpc.config.TRAIN_ITERS + 1):
        timer.start("train-iterations")
        engine.train()
        if use_pipeline:
            engine.zero_grad()
            _, _, train_loss = engine.execute_schedule(train_data_iter, return_output_label=False)
            engine.step()
        else:
            tokens, types, sentence_order, loss_mask, lm_labels, padding_mask = get_batch_for_sequence_parallel(
                trainloader
            )
            engine.zero_grad()
            lm_loss, sop_output = engine(tokens, padding_mask, types, lm_labels)
            train_loss = engine.criterion(lm_loss, sop_output, loss_mask, sentence_order)
            engine.backward(train_loss)
            engine.step()
        timer.stop("train-iterations", keep_in_history=True)

        if not gpc.is_initialized(ParallelMode.PIPELINE) or gpc.is_last_rank(ParallelMode.PIPELINE):
            accumulated_train_loss += train_loss

        lr_scheduler.step()

        if step % gpc.config.EVAL_INTERVAL == 0:
            engine.eval()

            for j in range(gpc.config.EVAL_ITERS):
                with torch.no_grad():
                    if use_pipeline:
                        _, _, eval_loss = engine.execute_schedule(
                            valid_data_iter, forward_only=True, return_output_label=False
                        )
                    else:
                        (
                            tokens,
                            types,
                            sentence_order,
                            loss_mask,
                            lm_labels,
                            padding_mask,
                        ) = get_batch_for_sequence_parallel(validloader)
                        lm_loss, sop_output = engine(tokens, padding_mask, types, lm_labels)
                        eval_loss = engine.criterion(lm_loss, sop_output, loss_mask, sentence_order)

                    if not gpc.is_initialized(ParallelMode.PIPELINE) or gpc.is_last_rank(ParallelMode.PIPELINE):
                        accumulated_eval_loss += eval_loss

            if not gpc.is_initialized(ParallelMode.PIPELINE) or gpc.is_last_rank(ParallelMode.PIPELINE):
                accumulated_eval_loss /= gpc.config.EVAL_ITERS
                accumulated_train_loss /= gpc.config.EVAL_INTERVAL

            timer_string = []
            for n, t in timer:
                timer_string.append(f"{n}: {t.get_history_mean()*1000:.5f}")
            timer_string = " | ".join(timer_string)
            lr = list(engine.optimizer.param_groups)[0]["lr"]
            loss_scale = engine.optimizer.optim.loss_scale.item()

            if gpc.is_initialized(ParallelMode.PIPELINE):
                ranks = [gpc.get_ranks_in_group(ParallelMode.PIPELINE)[-1]]
            else:
                ranks = [0]
            logger.info(
                f"Step {step} / {gpc.config.TRAIN_ITERS} | Train Loss: {accumulated_train_loss.item():.5g} "
                + f"| Eval Loss: {accumulated_eval_loss.item():.5g} "
                + f"| Loss Scale: {loss_scale}"
                + f"| Learning rate: {lr} | "
                + timer_string,
                ranks=ranks,
            )

            for n, t in timer:
                t.reset()
            accumulated_eval_loss.zero_()
            accumulated_train_loss.zero_()


if __name__ == "__main__":
    main()
