import argparse

import tensorrt_llm

from colossalai.inference.trtllm.model.llama.llama import LlamaEngineRunner, LlamaRunnerConfig

if __name__ == "__main__":
    args = LlamaRunnerConfig.add_cli_args(argparse.ArgumentParser()).parse_args()
    engine_runner = LlamaEngineRunner()
    runtime_rank = tensorrt_llm.mpi_rank()
    outputs = engine_runner.generate(**vars(args))
    if runtime_rank == 0:
        print("outputs: ", outputs)
