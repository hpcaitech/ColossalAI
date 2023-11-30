import argparse
import time

import torch
import torch.multiprocessing as mp
from tensorrt_llm.logger import logger

from colossalai.inference.trtllm.model.llama.llama import LlamaArgsConfig, LlamaEngineBuilder

if __name__ == "__main__":
    args = LlamaArgsConfig.add_cli_args(argparse.ArgumentParser()).parse_args()
    builder_config = LlamaArgsConfig.from_cli_args(args)
    builder = LlamaEngineBuilder()
    builder.set_config(builder_config)
    tik = time.time()
    if args.parallel_build and args.world_size > 1 and torch.cuda.device_count() >= args.world_size:
        logger.warning(
            f"Parallelly build TensorRT engines. Please make sure that all of the {args.world_size} GPUs are totally free."
        )
        mp.spawn(builder.build, nprocs=args.world_size, args=("llama",))
    else:
        args.parallel_build = False
        logger.info("Serially build TensorRT engines.")
        builder.build(0, "llama")

    tok = time.time()
    t = time.strftime("%H:%M:%S", time.gmtime(tok - tik))
    logger.info(f"Total time of building all {args.world_size} engines: {t}")
