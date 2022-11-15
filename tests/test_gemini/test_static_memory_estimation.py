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



GPT_BATCH_SIZE = 8
TM_BATCH_SIZE = 64


class MyTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.module_list = nn.ModuleList()
        for iii in range(8):
            self.module_list.append(nn.Linear(1024, 1024, bias=False))

    def forward(self, x):
        for mmm in self.module_list:
            x = mmm(x)
        return x


class BertLMModel(nn.Module):
    def __init__(self, hidden_size=512, num_layers=12, num_attention_heads=12, max_seq_len=1024, vocab_size=28996,
                 checkpoint=False):
        super().__init__()
        self.checkpoint = checkpoint
        self.model = BertLMHeadModel(BertConfig(n_embd=hidden_size, n_layer=num_layers,
                                                n_head=num_attention_heads, n_positions=max_seq_len, n_ctx=max_seq_len,
                                                vocab_size=vocab_size))
        if checkpoint:
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask):
        # Only return lm_logits
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=not self.checkpoint)[0]


class GPTLMModel(nn.Module):
    def __init__(self, hidden_size=768, num_layers=12, num_attention_heads=12, max_seq_len=1024, vocab_size=50257,
                 checkpoint=False):
        super().__init__()
        self.checkpoint = checkpoint
        self.model = GPT2LMHeadModel(GPT2Config(n_embd=hidden_size, n_layer=num_layers,
                                                n_head=num_attention_heads, n_positions=max_seq_len, n_ctx=max_seq_len,
                                                vocab_size=vocab_size))
        if checkpoint:
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask):
        # Only return lm_logits
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=not self.checkpoint)[0]


class GPTLMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def get_data(batch_size, seq_len, vocab_size):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=torch.cuda.current_device())
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def bert_base(checkpoint=False):
    return BertLMModel(hidden_size=512, num_layers=1, num_attention_heads=12, checkpoint=checkpoint)


def gpt2_medium(checkpoint=False):
    return GPTLMModel(hidden_size=1024, num_layers=12, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_xl(checkpoint=True):
    return GPTLMModel(hidden_size=1600, num_layers=48, num_attention_heads=32, checkpoint=checkpoint)


def gpt2_10b(checkpoint=True):
    return GPTLMModel(hidden_size=4096, num_layers=50, num_attention_heads=16, checkpoint=checkpoint)


def get_cpu_mem():
    return psutil.Process().memory_info().rss / 1024 ** 2


def get_gpu_mem():
    return torch.cuda.memory_allocated() / 1024 ** 2


def get_mem_info(prefix=''):
    return f'{prefix}GPU memory usage: {get_gpu_mem():.2f} MB, CPU memory usage: {get_cpu_mem():.2f} MB'


def get_tflops(model_numel, batch_size, seq_len, step_time):
    return model_numel * batch_size * seq_len * 8 / 1e12 / (step_time + 1e-12)


def main():
    PLACEMENT_POLICY = 'auto'
    disable_existing_loggers()
    colossalai.launch_from_torch(config={})
    pg = ProcessGroup()
    logger = get_dist_logger()

    logger.info(get_mem_info(), ranks=[0])
    # build model
    with ColoInitContext(device=get_current_device()):
        # model = bert_base(checkpoint=False)
        model = gpt2_medium(checkpoint=False)
        # model = MyTestModel()

    numel = sum([p.numel() for p in model.parameters()])
    logger.info(f'Model numel: {numel}', ranks=[0])

    cai_version = colossalai.__version__
    logger.info(f'using Colossal-AI version {cai_version}')
    if version.parse(cai_version) > version.parse("0.1.10"):
        from colossalai.gemini import ChunkManager, GeminiManager, search_chunk_configuration
        config_dict, _ = search_chunk_configuration(model, search_range_mb=1, search_interval_byte=100)
        # print(config_dict)
        chunk_manager = ChunkManager(config_dict,
                                     init_device=GeminiManager.get_default_device(PLACEMENT_POLICY))
        gemini_manager = GeminiManager(PLACEMENT_POLICY, chunk_manager, model, True)
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

    # data, mask = get_data(GPT_BATCH_SIZE, 512, 28996)
    data, mask = get_data(GPT_BATCH_SIZE, 1024, 50257)
    # data = torch.randn(int(TM_BATCH_SIZE), 1024, device=get_current_device())

    output = model(data, mask)
    # output = model(data)
    loss = torch.mean(output)
    model.backward(loss)

    cuda_model_data_list = np.array(model.gemini_manager._mem_stats_collector.model_data_list('cuda')) / 1024 ** 2
    print("cuda_model_data_list", len(cuda_model_data_list))
    # print(cuda_model_data_list)

    cuda_non_model_data_list = np.array(model.gemini_manager._mem_stats_collector.non_model_data_list('cuda')) / 1024 ** 2
    print("cuda_non_model_data_list", len(cuda_non_model_data_list))
    # print(cuda_non_model_data_list)

    overall_data_list = np.array(model.gemini_manager._mem_stats_collector.overall_mem_stats("cuda")) / 1024 ** 2
    print("_overall_cuda_list", len(overall_data_list))
    # print(overall_data_list)


if __name__ == '__main__':
    main()