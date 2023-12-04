import argparse
import time

import tensorrt_llm
import torch
from tensorrt_llm.logger import logger

from colossalai.inference.trtllm.model.llama.builder_and_runner import LlamaArgsConfig, LlamaEngineBuilder

if __name__ == "__main__":
    args = LlamaArgsConfig.add_args_argument(LlamaArgsConfig, argparse.ArgumentParser()).parse_args()
    builder_config = LlamaArgsConfig.init_from_args(args)
    builder = LlamaEngineBuilder()
    builder.set_config(builder_config)
    tik = time.time()
    rank = tensorrt_llm.mpi_rank()
    if args.parallel_build and args.world_size > 1 and torch.cuda.device_count() >= args.world_size:
        logger.warning(
            f"Parallelly build TensorRT engines. Please make sure that all of the {args.world_size} GPUs are totally free."
        )
        builder.build(rank, "llama")
    else:
        args.parallel_build = False
        logger.info("Serially build TensorRT engines.")
        builder.build(rank, "llama")

    tok = time.time()
    t = time.strftime("%H:%M:%S", time.gmtime(tok - tik))
    logger.info(f"Total time of building all {args.world_size} engines: {t}")
