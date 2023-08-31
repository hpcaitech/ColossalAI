import warnings

import torch
import torch.distributed as dist

HAS_AUTO_GPTQ = False
try:
    import auto_gptq
    HAS_AUTO_GPTQ = True
except ImportError:
    warnings.warn('please install auto-gptq from https://github.com/PanQiWei/AutoGPTQ')
    HAS_AUTO_GPTQ = False

from .cai_gptq import CaiQuantLinear
from .models import GPTQBloomConfig, GPTQLlamaConfig, reset_bloom_attention_params, reset_llama_attention_params

model_config_map = {
    "llama": GPTQLlamaConfig,
    "bloom": GPTQBloomConfig,
}
attention_proc_map = {
    "llama": reset_llama_attention_params,
    "bloom": reset_bloom_attention_params,
}
if HAS_AUTO_GPTQ:

    def get_module_by_name_prefix(model, module_name: str):
        for name, module in model.named_modules():
            if name.startswith(module_name):
                return module

    def split_column_copy(gptq_linear, cai_linear, tp_size=1, tp_rank=0, split_num=1):

        qweights = gptq_linear.qweight.split(gptq_linear.out_features // split_num, dim=-1)
        qzeros = gptq_linear.qzeros.split(gptq_linear.out_features // (32 // cai_linear.bits) // split_num, dim=-1)
        scales = gptq_linear.scales.split(gptq_linear.out_features // split_num, dim=-1)
        g_idx = gptq_linear.g_idx
        if gptq_linear.bias is not None:
            bias = gptq_linear.bias.split(gptq_linear.out_features // split_num, dim=-1)

        cai_split_out_features = cai_linear.outfeatures // split_num
        zero_split_block = cai_linear.outfeatures // (32 // cai_linear.bits) // split_num

        for i in range(split_num):
            cai_linear.qweight[:, i * cai_split_out_features:(i + 1) *
                               cai_split_out_features] = qweights[i][:, tp_rank * cai_split_out_features:(tp_rank + 1) *
                                                                     cai_split_out_features]
            cai_linear.qzeros[:, i * zero_split_block:(i + 1) *
                              zero_split_block] = qzeros[i][:,
                                                            tp_rank * zero_split_block:(tp_rank + 1) * zero_split_block]
            cai_linear.scales[:, i * cai_split_out_features:(i + 1) *
                              cai_split_out_features] = scales[i][:, tp_rank * cai_split_out_features:(tp_rank + 1) *
                                                                  cai_split_out_features]
            if cai_linear.bias is not None:
                cai_linear.bias[i * cai_split_out_features:(i + 1) *
                                cai_split_out_features] = bias[i][tp_rank * cai_split_out_features:(tp_rank + 1) *
                                                                  cai_split_out_features]

        cai_linear.g_idx.copy_(g_idx)

    def split_row_copy(gptq_linear, cai_linear, tp_size=1, tp_rank=0, split_num=1):

        qweights = gptq_linear.qweight.split(gptq_linear.in_features // split_num, dim=0)
        qzeros = gptq_linear.qzeros.split(gptq_linear.in_features // split_num, dim=0)
        scales = gptq_linear.scales.split(gptq_linear.in_features // split_num, dim=0)
        g_idxs = gptq_linear.g_idx.split(gptq_linear.in_features // split_num, dim=0)

        cai_split_in_features = cai_linear.infeatures // (32 // cai_linear.bits) // split_num
        zero_split_block = cai_linear.infeatures // cai_linear.groupsize // split_num
        idx_split_features = cai_linear.infeatures // split_num

        for i in range(split_num):
            cai_linear.qweight[i * cai_split_in_features:(i + 1) *
                               cai_split_in_features, :] = qweights[i][tp_rank * cai_split_in_features:(tp_rank + 1) *
                                                                       cai_split_in_features, :]
            cai_linear.qzeros[i * zero_split_block:(i + 1) *
                              zero_split_block, :] = qzeros[i][tp_rank * zero_split_block:(tp_rank + 1) *
                                                               zero_split_block, :]
            cai_linear.scales[i * zero_split_block:(i + 1) *
                              zero_split_block, :] = scales[i][tp_rank * zero_split_block:(tp_rank + 1) *
                                                               zero_split_block, :]
            cai_linear.g_idx[i * idx_split_features:(i + 1) *
                             idx_split_features] = g_idxs[i][tp_rank * idx_split_features:(tp_rank + 1) *
                                                             idx_split_features]
        if cai_linear.bias is not None:
            cai_linear.bias.copy_(gptq_linear.bias)

    def replace_autogptq_linear(model, tp_size=1, tp_rank=0, tp_group=None):

        def all_reduce_hook(cai_linear, input, output):
            dist.all_reduce(output, op=dist.ReduceOp.SUM, group=tp_group)
            if cai_linear.bias is not None:
                output.add_(cai_linear.bias)

        model_type_name = model.config.model_type

        gptq_model_config = model_config_map[model_type_name]
        layers = get_module_by_name_prefix(model.model, gptq_model_config.layer_blocks)

        for layer in layers:

            attention_proc_map[model_type_name](layer, tp_size=tp_size)
            for linear_name in gptq_model_config.linear_names[0]:
                gptq_linear = get_module_by_name_prefix(layer, linear_name)
                #column split copy
                cai_linear = CaiQuantLinear(
                    gptq_linear.bits,
                    gptq_linear.group_size,
                    gptq_linear.in_features,
                    gptq_linear.out_features // tp_size,
                    gptq_linear.bias is not None,
                    tp_size=tp_size,
                    tp_rank=tp_rank,
                )
                cai_linear.to(gptq_linear.qweight.device)
                if len(gptq_model_config.linear_names[0]) == 1:
                    split_column_copy(gptq_linear, cai_linear, tp_size=tp_size, tp_rank=tp_rank, split_num=3)
                else:
                    split_column_copy(gptq_linear, cai_linear, tp_size=tp_size, tp_rank=tp_rank, split_num=1)
                name1, name2 = linear_name.split(".")
                parent_module = get_module_by_name_prefix(layer, name1)
                setattr(parent_module, name2, cai_linear)

            for linear_name in gptq_model_config.linear_names[1]:
                gptq_linear = get_module_by_name_prefix(layer, linear_name)
                #row split copy
                cai_linear = CaiQuantLinear(gptq_linear.bits,
                                            gptq_linear.group_size,
                                            gptq_linear.in_features // tp_size,
                                            gptq_linear.out_features,
                                            gptq_linear.bias is not None,
                                            tp_size=tp_size,
                                            tp_rank=tp_rank,
                                            row_split=True)
                cai_linear.to(gptq_linear.qweight.device)
                split_row_copy(gptq_linear, cai_linear, tp_size=tp_size, tp_rank=tp_rank)

                if tp_size > 1:
                    cai_linear.register_forward_hook(all_reduce_hook)
                name1, name2 = linear_name.split(".")
                parent_module = get_module_by_name_prefix(layer, name1)
                setattr(parent_module, name2, cai_linear)

            for linear_name in gptq_model_config.linear_names[2]:
                gptq_linear = get_module_by_name_prefix(layer, linear_name)
                #column split copy
                cai_linear = CaiQuantLinear(
                    gptq_linear.bits,
                    gptq_linear.group_size,
                    gptq_linear.in_features,
                    gptq_linear.out_features // tp_size,
                    gptq_linear.bias is not None,
                    tp_size=tp_size,
                    tp_rank=tp_rank,
                )
                cai_linear.to(gptq_linear.qweight.device)
                split_column_copy(gptq_linear, cai_linear, tp_size=tp_size, tp_rank=tp_rank)
                name1, name2 = linear_name.split(".")
                parent_module = get_module_by_name_prefix(layer, name1)
                setattr(parent_module, name2, cai_linear)

            for linear_name in gptq_model_config.linear_names[3]:
                gptq_linear = get_module_by_name_prefix(layer, linear_name)
                #row split copy
                cai_linear = CaiQuantLinear(gptq_linear.bits,
                                            gptq_linear.group_size,
                                            gptq_linear.in_features // tp_size,
                                            gptq_linear.out_features,
                                            gptq_linear.bias is not None,
                                            tp_size=tp_size,
                                            tp_rank=tp_rank,
                                            row_split=True)
                cai_linear.to(gptq_linear.qweight.device)
                split_row_copy(gptq_linear, cai_linear, tp_size=tp_size, tp_rank=tp_rank)

                if tp_size > 1:
                    cai_linear.register_forward_hook(all_reduce_hook)
                name1, name2 = linear_name.split(".")
                parent_module = get_module_by_name_prefix(layer, name1)
                setattr(parent_module, name2, cai_linear)
