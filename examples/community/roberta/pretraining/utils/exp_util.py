import functools
import os
import shutil

import psutil
import torch

from colossalai.legacy.core import global_context as gpc


def logging(s, log_path, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(log_path, "a+") as f_log:
            f_log.write(s + "\n")


def get_logger(log_path, **kwargs):
    return functools.partial(logging, log_path=log_path, **kwargs)


def create_exp_dir(dir_path, scripts_to_save=None, debug=False):
    if debug:
        print("Debug Mode : no experiment dir created")
        return functools.partial(logging, log_path=None, log_=False)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    print("Experiment dir : {}".format(dir_path))
    if scripts_to_save is not None:
        script_path = os.path.join(dir_path, "scripts")
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        for script in scripts_to_save:
            dst_file = os.path.join(dir_path, "scripts", os.path.basename(script))
            shutil.copyfile(script, dst_file)

    return get_logger(log_path=os.path.join(dir_path, "log.txt"))


def get_cpu_mem():
    return psutil.Process().memory_info().rss / 1024**2


def get_gpu_mem():
    return torch.cuda.memory_allocated() / 1024**2


def get_mem_info(prefix=""):
    return f"{prefix}GPU memory usage: {get_gpu_mem():.2f} MB, CPU memory usage: {get_cpu_mem():.2f} MB"


def get_tflops(model_numel, batch_size, seq_len, step_time):
    return model_numel * batch_size * seq_len * 8 / 1e12 / (step_time + 1e-12)


def get_parameters_in_billions(model, world_size=1):
    gpus_per_model = world_size

    approx_parameters_in_billions = sum(
        [
            sum([p.ds_numel if hasattr(p, "ds_id") else p.nelement() for p in model_module.parameters()])
            for model_module in model
        ]
    )

    return approx_parameters_in_billions * gpus_per_model / (1e9)


def throughput_calculator(numel, args, config, iteration_time, total_iterations, world_size=1):
    gpus_per_model = 1
    batch_size = args.train_micro_batch_size_per_gpu
    batch_size * args.max_seq_length
    world_size / gpus_per_model
    approx_parameters_in_billions = numel
    elapsed_time_per_iter = iteration_time / total_iterations
    samples_per_second = batch_size / elapsed_time_per_iter

    # flops calculator
    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers
    vocab_size = config.vocab_size

    # General TFLOPs formula (borrowed from Equation 3 in Section 5.1 of
    # https://arxiv.org/pdf/2104.04473.pdf).
    # The factor of 4 is when used with activation check-pointing,
    # otherwise it will be 3.
    checkpoint_activations_factor = 4 if args.checkpoint_activations else 3
    flops_per_iteration = (
        24 * checkpoint_activations_factor * batch_size * args.max_seq_length * num_layers * (hidden_size**2)
    ) * (1.0 + (args.max_seq_length / (6.0 * hidden_size)) + (vocab_size / (16.0 * num_layers * hidden_size)))
    tflops = flops_per_iteration / (elapsed_time_per_iter * (10**12))
    return samples_per_second, tflops, approx_parameters_in_billions


def synchronize():
    if not torch.distributed.is_available():
        return
    if not torch.distributed.is_initialized():
        return
    world_size = torch.distributed.get_world_size()
    if world_size == 1:
        return
    torch.distributed.barrier()


def log_args(logger, args):
    logger.info("--------args----------")
    message = "\n".join([f"{k:<30}: {v}" for k, v in vars(args).items()])
    message += "\n"
    message += "\n".join([f"{k:<30}: {v}" for k, v in gpc.config.items()])
    logger.info(message)
    logger.info("--------args----------\n")
