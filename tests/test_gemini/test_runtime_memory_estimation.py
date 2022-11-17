import colossalai
import psutil
import torch
import torch.nn as nn
import numpy as np
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import HybridAdam
from transformers import GPT2Config, GPT2LMHeadModel, BertConfig, BertLMHeadModel
from time import time
from functools import partial
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.utils import get_current_device
from colossalai.nn.parallel import ZeroDDP
from colossalai.tensor import ProcessGroup

from packaging import version

from model_repo import SimpleNet, NestedNet, NoLeafModule, NetWithRepeatedlyComputedLayers


GPT_BATCH_SIZE = 8
TM_BATCH_SIZE = 64


def get_cpu_mem():
    return psutil.Process().memory_info().rss / 1024 ** 2


def get_gpu_mem():
    return torch.cuda.memory_allocated() / 1024 ** 2


def get_mem_info(prefix=''):
    return f'{prefix}GPU memory usage: {get_gpu_mem():.2f} MB, CPU memory usage: {get_cpu_mem():.2f} MB'


def main():
    PLACEMENT_POLICY = 'auto'
    disable_existing_loggers()
    colossalai.launch_from_torch(config={})
    pg = ProcessGroup()
    logger = get_dist_logger()

    logger.info(get_mem_info(), ranks=[0])

    # build model
    with ColoInitContext(device=get_current_device()):
        model = NetWithRepeatedlyComputedLayers(checkpoint=False)

    numel = sum([p.numel() for p in model.parameters()])
    logger.info(f'Model numel: {numel}', ranks=[0])

    cai_version = colossalai.__version__
    logger.info(f'using Colossal-AI version {cai_version}')
    if version.parse(cai_version) > version.parse("0.1.10"):
        from colossalai.gemini import ChunkManager, GeminiManager, search_chunk_configuration
        config_dict, _ = search_chunk_configuration(model, search_range_mb=1, search_interval_byte=100)
        chunk_manager = ChunkManager(config_dict,
                                     init_device=GeminiManager.get_default_device(PLACEMENT_POLICY))
        gemini_manager = GeminiManager(PLACEMENT_POLICY, chunk_manager)
        model = ZeroDDP(model, gemini_manager)
    elif version.parse(cai_version) <= version.parse("0.1.10") and version.parse(cai_version) >= version.parse("0.1.9"):
        from colossalai.gemini import ChunkManager, GeminiManager
        chunk_size = ChunkManager.search_chunk_size(model, 64 * 1024 ** 2, 32)
        chunk_manager = ChunkManager(chunk_size, pg, enable_distributed_storage=True,
                                     init_device=GeminiManager.get_default_device(PLACEMENT_POLICY))

    if version.parse(torch.__version__) > version.parse("0.1.11"):
        logger.error(f'{torch.__version__} may not supported, please use torch version 0.1.11')

    logger.info(get_mem_info(prefix='After init model, '), ranks=[0])

    logger.info(chunk_manager, ranks=[0])

    model.train()

    data = torch.rand(TM_BATCH_SIZE, 1024, device=get_current_device())
    # data = torch.randint(low=0, high=2048, size=(TM_BATCH_SIZE, 16), device=get_current_device())

    output = model(data)
    loss = torch.mean(output)
    model.backward(loss)

    cuda_non_model_data_list = np.array(model.gemini_manager._mem_stats_collector.non_model_data_list('cuda')) / 1024 ** 2
    print("cuda_non_model_data_list", len(cuda_non_model_data_list))
    print(cuda_non_model_data_list)

    d_res_file = open("non_model_res_dynamic_gemini.txt", "w", encoding="utf-8")
    for ddd in cuda_non_model_data_list:
        d_res_file.write(str(ddd) + "\n")
    d_res_file.close()


if __name__ == '__main__':
    main()